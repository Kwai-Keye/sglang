import functools
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from einops import rearrange
from transformers import PretrainedConfig
from transformers.activations import GELUActivation
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.utils import torch_int

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import QuickGELU, get_act_fn
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import apply_rotary_pos_emb
from sglang.srt.utils import add_prefix, get_bool_env_var, print_info_once

_TP_RANK_DEFAULT = 0
_TP_SIZE_DEFAULT = 1


@functools.lru_cache(maxsize=1)
def _fixed_tp_kwargs() -> dict:
    if get_bool_env_var("ENABLE_KEYE_VIT_TP"):
        return {
            "tp_rank": get_tensor_model_parallel_rank(),
            "tp_size": get_tensor_model_parallel_world_size(),
        }
    return {"tp_rank": _TP_RANK_DEFAULT, "tp_size": _TP_SIZE_DEFAULT}


try:
    from sglang.srt.utils import is_cuda

    if is_cuda():
        try:
            from sgl_kernel.flash_attn import flash_attn_varlen_func
        except ImportError:
            from flash_attn import flash_attn_varlen_func

            print_info_once(
                """
                Note: using fa2 from flash_attn pkg, the performance may be sub-optimal in Hopper.
                """
            )
    else:
        flash_attn_varlen_func = None
except ImportError:
    flash_attn_varlen_func = None
    print_info_once(
        """
        flash_attn_varlen_func is not available
        """
    )


class SiglipMLP(nn.Module):

    def __init__(
        self,
        config,
        act_layer: Type[nn.Module] = QuickGELU,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
            **_fixed_tp_kwargs(),
        )
        self.act = get_act_fn(config.hidden_act)
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
            **_fixed_tp_kwargs(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_parallel, _ = self.fc1(x)
        x_parallel = self.act(x_parallel)
        x, _ = self.fc2(x_parallel)
        return x


class KeyeVisionEmbeddings(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.cache_position_embedding = dict()
        self.cache_position_count = dict()
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.packing_position_embedding = nn.Embedding(32768, self.embed_dim)

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def interpolate_pos_encoding(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
        is_after_patchify: bool = False,
    ) -> torch.Tensor:

        num_positions = self.position_embedding.weight.shape[0]

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        if is_after_patchify:
            new_height = height
            new_width = width
        else:
            new_height = height // self.patch_size
            new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def fetch_position_embedding_lfu_cache(self, embeddings, h, w, max_cache: int = 20):
        grid = (h, w)
        if grid in self.cache_position_embedding:
            self.cache_position_count[grid] += 1
            return self.cache_position_embedding[grid]

        if len(self.cache_position_embedding) >= max_cache:
            min_hit_grid = min(
                self.cache_position_count,
                key=self.cache_position_count.get,
            )
            self.cache_position_count.pop(min_hit_grid)
            self.cache_position_embedding.pop(min_hit_grid)

        position_embedding = self.interpolate_pos_encoding(embeddings, h, w, True)
        self.cache_position_count[grid] = 1
        self.cache_position_embedding[grid] = position_embedding
        return position_embedding

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        position_ids: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[
            list[
                Union[
                    tuple[int, int, int],
                    list[tuple[int, int, int]],
                ]
            ]
        ] = None,
        interpolate_pos_encoding=False,
    ) -> torch.Tensor:
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.dim() == 5:
            if position_ids is None:
                raise ValueError(
                    "position_ids cannot be None when pixel_values.dim() is 5."
                )
            (
                batch_size,
                squence_len,
                channel,
                height,
                width,
            ) = pixel_values.shape
            target_dtype = self.patch_embedding.weight.dtype
            pixel_values = rearrange(pixel_values, "b l c h w -> (b l) c h w")
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
            embeddings = patch_embeds.flatten(-2).squeeze(-1)

            if interpolate_pos_encoding and image_grid_thw is not None:
                start = 0
                tmp_embeddings = list()
                for image_grid in image_grid_thw:
                    t, h, w = image_grid
                    end = start + t * h * w
                    image_embeddings = embeddings[start:end, :]
                    position_embedding = (
                        self.interpolate_pos_encoding(image_embeddings, h, w, True)
                        .squeeze(0)
                        .repeat(t, 1)
                    )
                    image_embeddings = image_embeddings + position_embedding
                    tmp_embeddings.append(image_embeddings)
                    start = end
                embeddings = torch.concat(tmp_embeddings, dim=0).unsqueeze(0)
            else:
                embeddings = embeddings + self.packing_position_embedding(position_ids)
            return embeddings
        else:
            raise ValueError(
                "Unsupported pixel_values dimension:"
                f" {pixel_values.dim()}. Expected 4 or 5."
            )


class SigLIPRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.rope_init()

    def rope_init(self):
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class KeyeSiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You
    Need' paper."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        hidden_size = config.hidden_size
        self.hidden_size = config.hidden_size
        tp_size = _fixed_tp_kwargs()["tp_size"]
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_attention_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            **_fixed_tp_kwargs(),
        )
        self.out_proj = RowParallelLinear(
            input_size=hidden_size,
            output_size=hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
            **_fixed_tp_kwargs(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        cu_seqlens: Optional[list[torch.Tensor]] = None,
        rope_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.q_size, self.kv_size, self.kv_size],
            dim=-1,
        )

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        batch_size = q.shape[0]

        if rope_emb is None:
            q = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
            k = k.view(
                *k.shape[:-1],
                self.num_kv_heads,
                self.head_dim,
            )
            v = v.view(
                *v.shape[:-1],
                self.num_kv_heads,
                self.head_dim,
            )
        else:
            if cu_seqlens is None:
                raise ValueError("cu_seqlens cannot be None when rope_emb is not None.")
            cos, sin = rope_emb
            q = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
            k = k.view(
                *k.shape[:-1],
                self.num_kv_heads,
                self.head_dim,
            )
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            v = v.view(
                *v.shape[:-1],
                self.num_kv_heads,
                self.head_dim,
            )

        q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])

        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
            softmax_scale=self.scale,
        )

        context_layer = rearrange(output, "(b s) ... -> b s ...", b=batch_size)

        context_layer = rearrange(context_layer, "b s h d -> b s (h d)").contiguous()

        output, _ = self.out_proj(context_layer)
        return output


class KeyeSiglipEncoderLayer(nn.Module):

    def __init__(
        self,
        config: Union[PretrainedConfig],
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.self_attn = KeyeSiglipAttention(
            config,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(
            config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        cu_seqlens: Optional[list[torch.Tensor]] = None,
        rope_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.FloatTensor]:

        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            cu_seqlens=cu_seqlens,
            rope_emb=rope_emb,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class KeyeSiglipEncoder(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.layers = nn.ModuleList(
            [
                KeyeSiglipEncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_idx}", prefix),
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.rotary_pos_emb = SigLIPRotaryEmbedding(
            head_dim // 2,
            config.rope_theta if hasattr(config, "rope_theta") else 10000,
        )

    @staticmethod
    def flatten_list(image_grid_thw):
        tmp_image_grid_thw = list()
        for image_grid in image_grid_thw:
            if isinstance(image_grid, list):
                tmp_image_grid_thw.extend(image_grid)
            else:
                tmp_image_grid_thw.append(image_grid)
        return tmp_image_grid_thw

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cu_seqlens: Optional[list[torch.Tensor]] = None,
        image_grid_thw: Optional[
            list[
                Union[
                    tuple[int, int, int],
                    list[tuple[int, int, int]],
                ]
            ]
        ] = None,
        height_position_ids: Optional[torch.Tensor] = None,
        width_position_ids: Optional[torch.Tensor] = None,
        use_rope: Optional[bool] = False,
        window_size: Optional[bool] = -1,
        vision_or_text: str = "vision",
    ) -> BaseModelOutput:
        device = inputs_embeds.device
        hidden_states = inputs_embeds
        if use_rope is True:
            flatten_image_grid_thw = self.flatten_list(image_grid_thw)

            if width_position_ids is None or height_position_ids is None:
                split_hids = list()
                split_wids = list()
                for t, h, w in flatten_image_grid_thw:
                    image_pids = torch.arange(t * h * w, device=device) % (h * w)
                    sample_hids = image_pids // w
                    sample_wids = image_pids % w
                    split_hids.append(sample_hids)
                    split_wids.append(sample_wids)
                width_position_ids = torch.concat(split_wids, dim=0)
                height_position_ids = torch.concat(split_hids, dim=0)

            pids = torch.stack(
                [height_position_ids, width_position_ids],
                dim=-1,
            )
            max_grid_size = pids.max() + 1
            rope_emb_max_grid = self.rotary_pos_emb(max_grid_size)
            rope_emb = rope_emb_max_grid[pids].flatten(1)
            rope_emb = rope_emb.repeat(1, 2)
            rope_emb = (rope_emb.cos(), rope_emb.sin())
        else:
            rope_emb = None

        attn_cu_seqlens = cu_seqlens
        hidden_states = inputs_embeds
        assert attention_mask is None

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
                cu_seqlens=attn_cu_seqlens,
                rope_emb=rope_emb,
            )
        return hidden_states


class KeyeSiglipVisionTransformer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = KeyeVisionEmbeddings(config)
        self.encoder = KeyeSiglipEncoder(
            config,
            quant_config=quant_config,
            prefix=add_prefix("encoder", prefix),
        )
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        attention_mask: Optional[torch.Tensor] = None,
        sample_indices: Optional[torch.Tensor] = None,
        image_indices: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        height_position_ids: Optional[torch.Tensor] = None,
        width_position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[list[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        vision_return_embed_list: Optional[bool] = False,
        image_grid_thw: Optional[
            list[
                Union[
                    tuple[int, int, int],
                    list[tuple[int, int, int]],
                ]
            ]
        ] = None,
        return_pooler_output: Optional[bool] = True,
        use_rope: Optional[bool] = False,
        window_size: Optional[bool] = -1,
    ) -> BaseModelOutputWithPooling:
        hidden_states = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
        )

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens,
            image_grid_thw=image_grid_thw,
            use_rope=use_rope,
            height_position_ids=height_position_ids,
            width_position_ids=width_position_ids,
            window_size=window_size,
            vision_or_text="vision",
        )

        last_hidden_state = self.post_layernorm(last_hidden_state)

        sample_hidden_state = list()
        if cu_seqlens is None:
            raise ValueError(
                "cu_seqlens cannot be None for "
                "SiglipVisionTransformer output processing."
            )
        for i in range(cu_seqlens.shape[0] - 1):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            tensor = last_hidden_state[:, start:end, :].squeeze(0)
            sample_hidden_state.append(tensor)

        return sample_hidden_state


class KeyeSiglipVisionModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.vision_model = KeyeSiglipVisionTransformer(
            config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.quant_config = quant_config

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_model.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.vision_model.embeddings.patch_embedding.weight.device

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        sample_indices: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        height_position_ids: Optional[torch.Tensor] = None,
        width_position_ids: Optional[torch.Tensor] = None,
        vision_return_embed_list: Optional[bool] = False,
        image_grid_thw: Optional[
            list[
                Union[
                    tuple[int, int, int],
                    list[tuple[int, int, int]],
                ]
            ]
        ] = None,
        cu_seqlens: Optional[list[torch.Tensor]] = None,
        return_pooler_output: Optional[bool] = True,
        use_rope: Optional[bool] = False,
        window_size: Optional[bool] = -1,
    ) -> BaseModelOutputWithPooling:

        return self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            position_ids=position_ids,
            vision_return_embed_list=vision_return_embed_list,
            image_grid_thw=image_grid_thw,
            sample_indices=sample_indices,
            cu_seqlens=cu_seqlens,
            return_pooler_output=return_pooler_output,
            use_rope=use_rope,
            window_size=window_size,
        )


class Projector(nn.Module):

    def __init__(
        self,
        text_config: PretrainedConfig,
        vision_config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.text_config = text_config
        self.vision_config = vision_config

        self.hidden_size = (
            self.vision_config.hidden_size
            * self.vision_config.spatial_merge_size
            * self.vision_config.spatial_merge_size
        )

        self.pre_norm = torch.nn.LayerNorm(self.hidden_size, eps=1e-05)
        self.act = GELUActivation()

        self.linear_1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("linear_1", prefix),
            **_fixed_tp_kwargs(),
        )
        self.linear_2 = RowParallelLinear(
            self.hidden_size,
            self.text_config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("linear_2", prefix),
            **_fixed_tp_kwargs(),
        )

    def forward(
        self,
        image_features: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        image_grid_thw: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        # TODO: can optimize when input is video
        assert isinstance(image_features, (list, tuple))
        processed_features = list()
        for image_feature, image_grid in zip(image_features, image_grid_thw):
            t, h, w = image_grid
            from einops import rearrange

            m1 = m2 = self.vision_config.spatial_merge_size
            image_feature = rearrange(
                image_feature,
                "(t h p1 w p2) d -> (t h w) (p1 p2 d)",
                t=t,
                h=h // m1,
                p1=m1,
                w=w // m2,
                p2=m2,
            )
            image_feature = self.pre_norm(image_feature)
            hidden_states, _ = self.linear_1(image_feature)
            hidden_states = self.act(hidden_states)
            hidden_states, _ = self.linear_2(hidden_states)
            processed_features.append(hidden_states)

        return processed_features
