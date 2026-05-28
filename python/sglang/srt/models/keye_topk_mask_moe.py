"""
KeyeTopKMaskMoe 模型实现 - sglang 版本
结合 Top-K Mask 稀疏注意力 + MoE (Mixture of Experts)
基于 Qwen3 MoE 架构，使用 KeyeTopKMask 稀疏注意力机制

继承关系:
- KeyeTopKMaskMoeDecoderLayer <- Qwen3MoeDecoderLayer  (替换 self_attn)
- KeyeTopKMaskMoeModel        <- Qwen3MoeModel          (传入 decoder_layer_type)
- KeyeVL2MoeForConditionalGeneration <- KeyeVL1_5ForConditionalGeneration
    (共用 __init__, get_image_feature, get_video_feature, forward;
     只覆盖 load_weights 以处理 MoE expert 权重 + sa_indexer 权重)
"""

import logging
from typing import Iterable, Optional, Tuple

import torch
from transformers import PretrainedConfig

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.keye_topk_mask import KeyeTopKMaskAttention
from sglang.srt.models.keye_qwen3 import KeyeVL1_5ForConditionalGeneration
from sglang.srt.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeModel
from sglang.srt.models.qwen3_vl_moe import load_fused_expert_weights
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.utils import add_prefix, is_cuda, is_hopper_with_cuda_12_3, print_info_once

_is_cuda = is_cuda()

logger = logging.getLogger(__name__)


class KeyeTopKMaskMoeDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
        )

        sa_config = getattr(config, "sa_config", None)
        if is_hopper_with_cuda_12_3() and sa_config is not None:
            head_dim = getattr(config, "head_dim", None)
            self.self_attn = KeyeTopKMaskAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                layer_id=layer_id,
                rope_theta=getattr(config, "rope_theta", 1000000),
                rope_scaling=getattr(config, "rope_scaling", None),
                head_dim=head_dim,
                max_position_embeddings=getattr(config, "max_position_embeddings", 32768),
                quant_config=quant_config,
                rms_norm_eps=config.rms_norm_eps,
                attention_bias=config.attention_bias,
                prefix=add_prefix("self_attn", prefix),
                alt_stream=alt_stream,
                sa_config=sa_config,
            )


class KeyeTopKMaskMoeModel(Qwen3MoeModel):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=KeyeTopKMaskMoeDecoderLayer,
        )

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embed_tokens


class KeyeVL2MoeForConditionalGeneration(KeyeVL1_5ForConditionalGeneration):
    """
    完整的 KeyeTopKMask MoE 多模态模型

    继承自 KeyeVL1_5ForConditionalGeneration，复用：
    - __init__: visual, mlp_AR, lm_head, logits_processor, pooler 构建
    - get_image_feature / get_video_feature: 多模态特征提取
    - forward: 前向传播

    只覆盖 load_weights，以额外处理：
    1. MoE expert 权重（expert_params_mapping）
    2. sa_indexer 权重（直接加载，跳过融合映射）
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        # 始终使用 KeyeTopKMaskMoeModel 作为语言模型
        # 语言模型内部会根据GPU架构选择合适的 decoder layer
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            language_model_cls=KeyeTopKMaskMoeModel,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        num_experts = self.config.num_experts

        vision_stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        vision_ignore_names = [
            "head.attention",
            "head.mlp",
            "head.layernorm",
            "head.probe",
        ]

        # Cache params_dict
        if not hasattr(self, "_cached_params_dict"):
            self._cached_params_dict = dict(self.named_parameters())
        params_dict = self._cached_params_dict

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # sa_indexer 权重：直接加载，跳过所有融合映射
            if "sa_indexer" in name:
                if name not in params_dict:
                    # print_info_once(f"Skipping sa_indexer weight (not in model): {name}")
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            # --- Vision weights (visual.*, mlp_AR.*) ---
            if name.startswith("visual.") or name.startswith("mlp_AR."):
                matched = False
                for param_name, weight_name, shard_id in vision_stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    if not name.startswith("visual."):
                        continue
                    new_name = name.replace(weight_name, param_name)
                    if new_name.endswith(".bias") and new_name not in params_dict:
                        matched = True
                        break
                    if new_name not in params_dict:
                        continue
                    param = params_dict[new_name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    matched = True
                    break

                if matched:
                    continue

                if name.endswith(".bias") and name not in params_dict:
                    continue
                new_name = maybe_remap_kv_scale_name(name, params_dict)
                if new_name is None:
                    continue
                if any(ign in new_name for ign in vision_ignore_names):
                    continue
                if new_name not in params_dict:
                    continue
                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            # --- LLM weights (model.*, lm_head.*) ---
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        loaded_weight = loaded_weight.transpose(-1, -2)
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped, params_dict, loaded_weight[0], "w1", num_experts
                            )
                            load_fused_expert_weights(
                                name_mapped, params_dict, loaded_weight[1], "w3", num_experts
                            )
                        else:
                            load_fused_expert_weights(
                                name_mapped, params_dict, loaded_weight, shard_id, num_experts
                            )
                    else:
                        if name_mapped.endswith(ignore_suffixes) and name_mapped not in params_dict:
                            continue
                        param = params_dict[name_mapped]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, name_mapped, shard_id=shard_id, expert_id=expert_id)
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        continue
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


EntryClass = [KeyeVL2MoeForConditionalGeneration]
