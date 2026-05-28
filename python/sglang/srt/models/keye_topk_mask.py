"""
KeyeTopKMask 模型实现 - sglang 版本
基于 Qwen3 架构,使用 Top-K Mask 稀疏注意力机制优化推理性能
"""
import os
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.attention.keye_topk.keye_indexer import KeyeIndexer
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.keye_qwen3 import KeyeVL1_5ForConditionalGeneration
from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP, Qwen2Model
from sglang.srt.models.qwen3 import Qwen3DecoderLayer
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, is_hopper_with_cuda_12_3, print_info_once

_is_cuda = is_cuda()

logger = logging.getLogger(__name__)

class KeyeTopKMaskAttention(nn.Module):
    """
    Top-K Mask Sparse Attention for Keye model (sglang version).
    
    使用 Stage 1 训练的 Indexer 的 top-k 结果来构建注意力掩码，
    然后应用带掩码的 eager attention 进行稀疏推理。
    
    核心思想:
      1. 正常计算 Q*K^T 注意力分数
      2. 使用 Indexer 的 top-k 索引来掩码非选中的 KV 位置 (-inf)
      3. 应用 softmax 并与 V 相乘
    
    Args:
        hidden_size: 隐藏层大小
        num_heads: 注意力头数量
        num_kv_heads: KV 头数量（GQA）
        layer_id: 层编号
        rope_theta: RoPE theta 参数
        rope_scaling: RoPE scaling 配置
        head_dim: 每个头的维度
        max_position_embeddings: 最大位置嵌入
        quant_config: 量化配置
        rms_norm_eps: RMS norm epsilon
        attention_bias: 是否使用 attention bias
        prefix: 权重前缀
        alt_stream: 可选的 CUDA stream
        sa_config: 稀疏注意力配置
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = True,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        sa_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        
        # 处理 TP - 使用 attention-specific TP functions
        from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        
        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            self.head_dim = hidden_size // self.total_num_heads
        
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.num_key_value_groups = self.total_num_heads // self.total_num_kv_heads
        self.is_causal = True
        self.attention_dropout = 0.0  # Inference mode
        self.rope_scaling = rope_scaling
        
        # Q, K, V, O 投影层
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )
        
        # Q/K Normalization
        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_global_server_args().rl_on_policy_target is not None
            else {}
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        
        # RoPE
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        
        # Radix Attention (用于标准的 KV cache 管理)
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.alt_stream = alt_stream
        
        # === Indexer for Sparse Attention ===
        self.sa_indexer: Optional[KeyeIndexer] = None
        self.sa_topk = 2048
        
        # 从 sa_config 初始化 Indexer
        if sa_config is not None:
            indexer_num_heads = sa_config.get('indexer_num_heads', 4)
            indexer_head_dim = sa_config.get('indexer_head_dim', 128)
            topk = sa_config.get('topk', 2048)
            self.sa_topk = topk
            
            # 提取 mrope_section
            mrope_section = [16, 24, 24]  # 默认值
            if rope_scaling is not None and isinstance(rope_scaling, dict):
                mrope_section = rope_scaling.get('mrope_section', mrope_section)
            
            self.sa_indexer = KeyeIndexer(
                hidden_size=hidden_size,
                num_heads=indexer_num_heads,
                head_dim=indexer_head_dim,
                topk=topk,
                mrope_section=mrope_section,
                rope_theta=rope_theta,
                main_head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                scale_fmt="ue8m0",
                block_size=128,
                layer_id=layer_id,
                alt_stream=alt_stream,
                quant_config=quant_config,
                prefix=add_prefix("sa_indexer", prefix),
            )

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Q/K normalization with optional overlap optimization."""
        # overlap qk norm in CUDA graph mode
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            with torch.cuda.stream(self.alt_stream):
                k_by_head = k.reshape(-1, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
            current_stream.wait_stream(self.alt_stream)
        else:
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            k_by_head = k.reshape(-1, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """
        Forward pass with Top-K Mask sparse attention.
        
        处理两种情况：
        1. Prefill (q_len > 1): 使用 Indexer 计算 top-k，应用稀疏 attention
        2. Decode (q_len == 1): 使用缓存的 Indexer K 计算 top-k，应用稀疏 attention
        
        注意：当前版本暂时不支持 topk_mask，需要后续扩展 RadixAttention
        
        Args:
            positions: Position IDs (可能是 3D [3, B, N] 用于 mrope)
            hidden_states: Input tensor, shape [B, q_len, hidden_size] or [total_tokens, hidden_size]
            forward_batch: Forward batch info (包含 KV cache 和 position IDs)
            
        Returns:
            attn_output: Same shape as input hidden_states
        """
        # ---- 1. Compute Q, K, V ----
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Apply Q/K normalization
        q, k = self._apply_qk_norm(q, k)
        
        # ---- 2. Apply RoPE ----
        q, k = self.rotary_emb(positions, q, k)
        
        # ---- 3. 运行 Indexer 获取 top-k (如果启用) ----
        topk_indices = None
        if self.sa_indexer is not None:
            topk_indices = self.sa_indexer(
                x=hidden_states,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=self.layer_id,
            )
        
        # ---- 4. 使用 RadixAttention (通过kwargs传递topk_indices) ----
        # topk_indices will be handled by KeyeSparseAttnBackend
        attn_output = self.attn(q, k, v, forward_batch, topk_indices=topk_indices)
        
        # ---- 5. Output Projection ----
        output, _ = self.o_proj(attn_output)
        
        return output


class KeyeTopKMaskDecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        config,
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



class KeyeTopKMaskModel(Qwen2Model):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        alt_stream = torch.cuda.Stream() if _is_cuda else None
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=KeyeTopKMaskDecoderLayer,
            alt_stream=alt_stream,
        )


class KeyeVL2ForConditionalGeneration(KeyeVL1_5ForConditionalGeneration):
    """
    完整的 KeyeTopKMask 多模态模型

    继承自 KeyeVL1_5ForConditionalGeneration，通过传入 language_model_cls 参数
    来使用 KeyeTopKMaskModel 作为语言模型主干

    架构:
    - visual: KeyeSiglipVisionModel (视觉编码器) - 继承
    - mlp_AR: Projector (视觉→文本投影) - 继承
    - model: KeyeTopKMaskModel (LLM 主干) - 通过 language_model_cls 指定
    - lm_head: 语言建模头 - 继承
    - 多模态处理逻辑 - 继承
    """


    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        # 始终使用 KeyeTopKMaskModel 作为语言模型
        # 语言模型内部会根据GPU架构选择合适的 decoder layer
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            language_model_cls=KeyeTopKMaskModel,
        )
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        权重加载，特殊处理 sa_indexer 的权重
        
        处理两种情况：
        1. checkpoint 中没有 sa_indexer 权重 -> 跳过，使用随机初始化
        2. checkpoint 中有 sa_indexer 权重 -> 正常加载
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # 特殊处理 sa_indexer 权重：直接加载，不进行融合映射
            if "sa_indexer" in name:
                if name not in params_dict:
                    # print_info_once(f"Skipping sa_indexer weight (not in model): {name}")
                    continue
                # 直接加载，跳过 stacked_params_mapping
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "visual" in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                
                # 检查参数是否存在
                if name not in params_dict:
                    print_info_once(f"Warning: parameter not found in model: {name}")
                    continue
                    
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                vision_stacked_params_mapping = [
                    ("qkv_proj", "q_proj", "q"),
                    ("qkv_proj", "k_proj", "k"),
                    ("qkv_proj", "v_proj", "v"),
                ]
                ignore_names = [
                    "head.attention",
                    "head.mlp",
                    "head.layernorm",
                    "head.probe",
                ]
                for (
                    param_name,
                    weight_name,
                    shard_id,
                ) in vision_stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    
                    # 检查参数是否存在
                    if name not in params_dict:
                        print_info_once(f"Warning: parameter not found in model: {name}")
                        continue
                        
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    if any(ignore_name in name for ignore_name in ignore_names):
                        continue
                    
                    # 检查参数是否存在，不存在则跳过（而不是报错）
                    if name not in params_dict:
                        print_info_once(f"Warning: parameter not found in model: {name}")
                        continue
                        
                    param = params_dict[name]
                    weight_loader = getattr(
                        param,
                        "weight_loader",
                        default_weight_loader,
                    )
                    weight_loader(param, loaded_weight)
    # 其他方法（pad_input_ids, get_image_feature, get_video_feature,
    # get_input_embeddings, forward）都从父类继承


# Register the model
EntryClass = [KeyeVL2ForConditionalGeneration]
