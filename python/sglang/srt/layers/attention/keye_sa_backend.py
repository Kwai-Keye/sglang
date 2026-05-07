"""Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

KeyeSA Backend — Keye Sparse Attention with GQA support (FlashAttn-based).

This backend is the only sparse attention backend for Keye models.
It uses flash_attn_with_kvcache paged attention for decode and a sparse kernel
for prefill, while keeping a CUDA-graph-safe metadata pattern and exposing
get_indexer_metadata() so that the shared KeyeIndexer can produce the
per-decode page table.

CUDA Graph pattern:
  - init_cuda_graph_state         → pre-allocate fixed-address buffers
  - init_forward_metadata_capture → fill buffers for each graph bs; store metadata
  - init_forward_metadata_replay  → copy new values into those same buffers in-place
  - get_indexer_metadata          → wrap current metadata as KeyeSAIndexerMetadata
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.attention.nsa.utils import compute_nsa_seqlens
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

try:
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    _has_flash_attn = True
except ImportError:
    _has_flash_attn = False

from effective_kernels.ops import sparse_attention_forward, topk_block_unique

from sglang.srt.layers.attention.nsa.transform_index import (
    transform_index_page_table_decode,
    transform_index_page_table_prefill,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
    """Compute cumulative sequence lengths (prefix-sum with leading zero)."""
    assert seqlens.dtype == torch.int32
    return torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )


def get_keye_sa_topk(config) -> int:
    assert config.sa_config is not None
    return config.sa_config["topk"]


# ---------------------------------------------------------------------------
# KeyeSAAttnMetadata — mutable buffers pre-allocated for CUDA graph safety
# ---------------------------------------------------------------------------

@dataclass
class KeyeSAAttnMetadata:
    """
    Mutable metadata buffers for KeyeSparseAttnBackend.

    NOT frozen so that init_forward_metadata_replay_cuda_graph can update
    fields via copy_() without breaking CUDA-graph-captured tensor addresses.

    Field names mirror NSAMetadata for consistency.
    """
    page_size: int

    # Standard attention fields
    cache_seqlens_int32: torch.Tensor   # [B] int32 — per-request KV lengths
    max_seq_len_q: int
    max_seq_len_k: int
    cu_seqlens_q: torch.Tensor          # [B+1] int32 cumulative Q lengths
    cu_seqlens_k: torch.Tensor          # [B+1] int32 cumulative K lengths
    # page-level indices (page_size=self.page_size); used by dense forward + indexer
    real_page_table: torch.Tensor
    # token-level slot indices (page_size=1); used by sparse transform_index
    page_table_1: torch.Tensor

    # Sparse-attention fields (clipped to topk)
    sa_cache_seqlens_int32: torch.Tensor   # [total_q] int32 — clipped to sa_topk
    sa_cu_seqlens_q: torch.Tensor          # arange(0, len(sa_cu_seqlens_k))
    sa_cu_seqlens_k: torch.Tensor          # cumsum of sa_cache_seqlens_int32
    sa_seqlens_expanded: torch.Tensor      # per-query expanded seqlens (prefill causal)
    sa_extend_seq_lens_list: List[int]     # extend lengths per request (CPU list)
    sa_max_seqlen_q: int = 1              # always 1 for decode; varies for prefill

    paged_mqa_schedule_metadata: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# KeyeSAIndexerMetadata — indexer-facing view (implements BaseIndexerMetadata)
# ---------------------------------------------------------------------------

from sglang.srt.layers.attention.nsa.nsa_indexer import BaseIndexerMetadata


@dataclass(frozen=True)
class KeyeSAIndexerMetadata(BaseIndexerMetadata):
    """Indexer-facing view of KeyeSAAttnMetadata.

    Implements BaseIndexerMetadata so that KeyeIndexer can call
    get_indexer_metadata() on KeyeSparseAttnBackend with the same protocol
    expected by the shared indexer interface.
    """
    attn_metadata: "KeyeSAAttnMetadata"
    paged_mqa_schedule_metadata: Optional[torch.Tensor] = None
    deterministic_topk: bool = False

    def get_seqlens_int32(self) -> torch.Tensor:
        return self.attn_metadata.cache_seqlens_int32

    def get_page_table_64(self) -> torch.Tensor:
        return self.attn_metadata.real_page_table

    def get_page_table_1(self) -> torch.Tensor:
        return self.attn_metadata.page_table_1

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self.attn_metadata.sa_seqlens_expanded

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        forward_batch: ForwardBatch,
        ks: Optional[torch.Tensor] = None,
        ke_offset: Optional[torch.Tensor] = None,
        context_length: int = 0,
    ) -> torch.Tensor:
        """Apply causal mask, then select top-k and return relative (or absolute) indices.

        This method owns the full masking + topk pipeline so callers do not need
        to perform ``masked_fill_`` before calling.

        The valid-K-length per query (``ke_offset``) defaults to
        ``attn_metadata.sa_seqlens_expanded`` (mirrors NSA's ``nsa_seqlens_expanded``):
          - prefill: per-extend-token causal length  (``ke - ks``, i.e. ke_offset)
          - decode:  full KV length per request      (== cache_seqlens_int32)
        Callers can override via the ``ke_offset`` argument, which is required for
        chunked prefill where only a slice of ``sa_seqlens_expanded`` applies.

        Args:
            logits:    [total_q, total_k] float32 — raw logits from fp8_mqa_logits /
                       fp8_paged_mqa_logits (may contain unmasked padding values).
            topk:      number of positions to select.
            ks:        [total_q] int32 — per-query K start offset (ragged/prefill).
                       When provided output indices are relative (idx - ks).
                       When None (decode) output indices are absolute.
            ke_offset: [total_q] int32 — per-query valid K length override.
                       When None, falls back to ``self.get_seqlens_expanded()``.
        Returns:
            [total_q, topk] int32, invalid positions filled with -1.
        """
        device = logits.device
        total_q = logits.shape[0]
        total_k = logits.shape[1]
        # ke_offset: valid K length per query. Use the caller-supplied slice for
        # chunked prefill; otherwise read from metadata (full tensor).
        if ke_offset is None:
            ke_offset = self.get_seqlens_expanded()

        if self.deterministic_topk:
            from flashinfer import top_k_ragged_transform, top_k_page_table_transform
            forward_mode = forward_batch.forward_mode
            if forward_mode.is_extend_without_speculative():
                ragged_offsets = forward_batch.attn_backend.get_ragged_offset(total_q)
                return top_k_ragged_transform(
                    logits[:, :context_length].contiguous(),
                    ragged_offsets,
                    ke_offset,
                    topk,
                    row_starts=ks,
                    deterministic=True,
                    dsa_graph_safe=True,
            )
            elif forward_mode.is_decode_or_idle() or forward_mode.is_draft_extend() or forward_mode.is_target_verify():
                return top_k_page_table_transform(
                    logits[:, :context_length].contiguous(),
                    self.attn_metadata.page_table_1,
                    ke_offset,
                    topk,
                    deterministic=True,
                    dsa_graph_safe=True,
                )

        # Fast path: fast_topk_v2 handles masking internally.
        if topk == 2048:
            from sgl_kernel import fast_topk_v2
            return fast_topk_v2(logits, ke_offset, topk, row_starts=ks)

        # Slow path: apply causal mask then use torch.topk.
        j = torch.arange(total_k, device=device, dtype=torch.int32).unsqueeze(0)
        if ks is not None:
            # ragged/prefill: mask columns outside [ks, ks + ke_offset)
            logits.masked_fill_(
                (j < ks.unsqueeze(1)) | (j >= (ks + ke_offset).unsqueeze(1)),
                torch.finfo(logits.dtype).min,
            )
        else:
            # decode: mask columns >= ke_offset (== seqlens_32)
            logits.masked_fill_(
                j >= ke_offset.unsqueeze(1),
                torch.finfo(logits.dtype).min,
            )

        actual_topk = min(topk, total_k)
        topk_values, topk_indices = torch.topk(logits, actual_topk, dim=-1)
        topk_indices = topk_indices.to(torch.int32)

        invalid = topk_values == torch.finfo(logits.dtype).min
        if ks is not None:
            rel_indices = (topk_indices - ks.unsqueeze(1)).masked_fill(invalid, -1)
        else:
            rel_indices = topk_indices.masked_fill(invalid, -1)

        if actual_topk == topk:
            return rel_indices

        result = torch.full((total_q, topk), -1, device=device, dtype=torch.int32)
        result[:, :actual_topk] = rel_indices
        return result


# ---------------------------------------------------------------------------
# KeyeSparseAttnBackend
# ---------------------------------------------------------------------------

class KeyeSparseAttnBackend(AttentionBackend):
    """Attention backend for Keye Sparse Attention with GQA support.

    Decode path: flash_attn_with_kvcache with a sparse page table derived from
                 topk_indices produced by KeyeIndexer.
    Prefill path: sparse_attention_forward (effective_kernels).
    """

    def __init__(
        self,
        model_runner: "ModelRunner",
        skip_prefill: bool = False,
        speculative_step_id: int = 0,
        speculative_num_steps: int = 0,
    ):
        super().__init__()
        self.forward_metadata: Optional[KeyeSAAttnMetadata] = None
        self.device = model_runner.device
        assert isinstance(model_runner.page_size, int)
        self.page_size = model_runner.page_size
        self.max_context_len = model_runner.model_config.context_len

        self.num_q_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        if hasattr(model_runner.model_config.hf_config, "num_key_value_heads"):
            self.num_kv_heads = (
                model_runner.model_config.hf_config.num_key_value_heads
                // get_attention_tp_size()
            )
        else:
            self.num_kv_heads = self.num_q_heads

        assert self.num_q_heads % self.num_kv_heads == 0, (
            f"num_q_heads ({self.num_q_heads}) must be divisible by "
            f"num_kv_heads ({self.num_kv_heads})"
        )
        self.num_kv_groups = self.num_q_heads // self.num_kv_heads
        self.head_dim = model_runner.model_config.head_dim
        self.sa_topk = get_keye_sa_topk(model_runner.model_config.hf_config)

        self.num_splits = (
            1 if model_runner.server_args.enable_deterministic_inference else 0
        )

        assert model_runner.req_to_token_pool is not None
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.max_model_len = model_runner.req_to_token_pool.max_context_len

        self._arange_buf = torch.arange(16384, device=self.device, dtype=torch.int32)

        # Speculative decoding
        self.speculative_step_id = speculative_step_id
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )

        # CUDA graph metadata: keyed by batch_size
        self.decode_cuda_graph_metadata: Dict[int, KeyeSAAttnMetadata] = {}

        self.deterministic_topk = get_global_server_args().deterministic_topk
        assert (not self.deterministic_topk) or (self.sa_topk <= 2048)

        self.zero_ragged_offset = None
        if self.deterministic_topk:
            chunked_prefill_size = get_global_server_args().chunked_prefill_size
            ragged_offsets_size = chunked_prefill_size if chunked_prefill_size is not None else 16384
            self.zero_ragged_offset = torch.zeros((ragged_offsets_size,), device=self.device, dtype=torch.int32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_device_int32_arange(self, l: int) -> torch.Tensor:
        if l > len(self._arange_buf):
            next_pow_of_2 = 1 << (l - 1).bit_length()
            self._arange_buf = torch.arange(
                next_pow_of_2, device=self.device, dtype=torch.int32
            )
        return self._arange_buf[:l]

    def get_ragged_offset(self, l:int) -> torch.Tensor:
        if l > len(self.zero_ragged_offset) or self.zero_ragged_offset is None:
            next_pow_of_2 = 1 << (l - 1).bit_length()
            self.zero_ragged_offset = torch.zeros(
                (next_pow_of_2,), device=self.device, dtype=torch.int32
            )
        return self.zero_ragged_offset[:l] 

    def _transform_table_1_to_real(self, page_table: torch.Tensor) -> torch.Tensor:
        page_size = self.page_size
        if page_size == 1:
            return page_table
        max_seqlen_k = page_table.shape[1]
        strided_indices = torch.arange(
            0, max_seqlen_k, page_size, device=page_table.device, dtype=torch.int32
        )
        return page_table[:, strided_indices] // page_size

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ------------------------------------------------------------------
    # get_indexer_metadata
    # ------------------------------------------------------------------

    def get_indexer_metadata(
        self,
        layer_id: int,
        forward_batch: "ForwardBatch",
    ) -> KeyeSAIndexerMetadata:
        return KeyeSAIndexerMetadata(
            attn_metadata=self.forward_metadata,
            paged_mqa_schedule_metadata=self.forward_metadata.paged_mqa_schedule_metadata,
            deterministic_topk=self.deterministic_topk,
        )

    # ------------------------------------------------------------------
    # init_forward_metadata — eager (non-graph) path
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize metadata for a forward pass (non-CUDA-graph path)."""
        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        if forward_batch.forward_mode.is_target_verify():
            draft_token_num = self.speculative_num_draft_tokens or 0
        else:
            draft_token_num = 0

        cache_seqlens_int32 = (forward_batch.seq_lens + draft_token_num).to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
        assert forward_batch.seq_lens_cpu is not None
        max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item()) + draft_token_num

        page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]

        if forward_batch.forward_mode.is_decode_or_idle():
            extend_seq_lens_cpu = [1] * batch_size
            max_seqlen_q = 1
            cu_seqlens_q = self.get_device_int32_arange(batch_size + 1)
            seqlens_expanded = cache_seqlens_int32
        elif forward_batch.forward_mode.is_extend_without_speculative():
            assert (
                forward_batch.extend_seq_lens_cpu is not None
                and forward_batch.extend_seq_lens is not None
            )
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            if any(forward_batch.extend_prefix_lens_cpu):
                max_seqlen_q = max(extend_seq_lens_cpu)
                cu_seqlens_q = compute_cu_seqlens(
                    forward_batch.extend_seq_lens.to(torch.int32)
                )
            else:
                max_seqlen_q = max_seqlen_k
                cu_seqlens_q = cu_seqlens_k

            seqlens_expanded = torch.cat(
                [
                    torch.arange(
                        kv_len - qo_len + 1,
                        kv_len + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    for qo_len, kv_len in zip(
                        forward_batch.extend_seq_lens_cpu,
                        forward_batch.seq_lens_cpu.tolist(),
                        strict=True,
                    )
                ]
            )
        else:
            raise ValueError(f"Unsupported forward_mode: {forward_batch.forward_mode}")

        sa_cache_seqlens_int32 = compute_nsa_seqlens(
            original_seq_lens=seqlens_expanded,
            nsa_index_topk=self.sa_topk,
        )
        sa_cu_seqlens_k = compute_cu_seqlens(sa_cache_seqlens_int32)
        sa_cu_seqlens_q = self.get_device_int32_arange(len(sa_cu_seqlens_k))

        page_table_1 = page_table

        paged_mqa_schedule_metadata = None
        if forward_batch.forward_mode.is_decode_or_idle():
            try:
                import deep_gemm

                paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    cache_seqlens_int32, 64, deep_gemm.get_num_sms()
                )
            except (ImportError, ModuleNotFoundError):
                paged_mqa_schedule_metadata = None

        self.forward_metadata = KeyeSAAttnMetadata(
            page_size=self.page_size,
            cache_seqlens_int32=cache_seqlens_int32,
            max_seq_len_q=max_seqlen_q,
            max_seq_len_k=max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            real_page_table=self._transform_table_1_to_real(page_table),
            page_table_1=page_table_1,
            sa_cache_seqlens_int32=sa_cache_seqlens_int32,
            sa_cu_seqlens_q=sa_cu_seqlens_q,
            sa_cu_seqlens_k=sa_cu_seqlens_k,
            sa_seqlens_expanded=seqlens_expanded,
            sa_extend_seq_lens_list=list(extend_seq_lens_cpu),
            sa_max_seqlen_q=max_seqlen_q,
            paged_mqa_schedule_metadata=paged_mqa_schedule_metadata,
        )

    # ------------------------------------------------------------------
    # CUDA graph state management
    # ------------------------------------------------------------------

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Pre-allocate fixed-address buffers for CUDA graph capture."""
        max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size

        self._graph_cache_seqlens = torch.ones(max_bs, dtype=torch.int32, device=self.device)
        self._graph_cu_seqlens_q = torch.arange(max_bs + 1, dtype=torch.int32, device=self.device)
        self._graph_cu_seqlens_k = torch.zeros(max_bs + 1, dtype=torch.int32, device=self.device)
        # real_page_table: page-level (page_size=self.page_size)
        self._graph_real_page_table = torch.zeros(
            max_bs, max_num_pages, dtype=torch.int32, device=self.device
        )
        # page_table_1: token-level (page_size=1)
        self._graph_page_table_1 = torch.zeros(
            max_bs, self.max_context_len, dtype=torch.int32, device=self.device
        )
        # Sparse fields (total_q = max_bs for decode)
        self._graph_sa_cache_seqlens = torch.ones(max_bs, dtype=torch.int32, device=self.device)
        self._graph_sa_cu_seqlens_k = torch.zeros(max_bs + 1, dtype=torch.int32, device=self.device)
        self._graph_sa_cu_seqlens_q = torch.arange(max_bs + 1, dtype=torch.int32, device=self.device)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info=None,
    ):
        """Fill graph buffers for batch size `bs` and store the metadata."""
        assert forward_mode.is_decode_or_idle()

        max_num_pages = self._graph_real_page_table.shape[1]
        max_len = int(seq_lens[:bs].max().item())

        # real_page_table (page-level)
        cache_seqlens_buf = self._graph_cache_seqlens[:bs]
        cache_seqlens_buf.copy_(seq_lens[:bs].to(torch.int32))
        cu_seqlens_q_buf = self._graph_cu_seqlens_q[: bs + 1]
        cu_seqlens_k_buf = self._graph_cu_seqlens_k[: bs + 1]
        cu_seqlens_k_buf[1:].copy_(
            torch.cumsum(cache_seqlens_buf, dim=0, dtype=torch.int32)
        )
        num_pages = (max_len + self.page_size - 1) // self.page_size
        real_page_table_buf = self._graph_real_page_table[:bs]
        strided = torch.arange(0, max_len, self.page_size, device=self.device, dtype=torch.int64)
        real_page_table_buf[:, :num_pages].copy_(
            self.req_to_token[req_pool_indices[:bs], :max_len][:, strided] // self.page_size
        )
        if num_pages < max_num_pages:
            real_page_table_buf[:, num_pages:].zero_()

        # page_table_1 (token-level)
        page_table_1_buf = self._graph_page_table_1[:bs, :max_len]
        page_table_1_buf.copy_(self.req_to_token[req_pool_indices[:bs], :max_len].to(torch.int32))
        if max_len < self._graph_page_table_1.shape[1]:
            self._graph_page_table_1[:bs, max_len:].zero_()

        # Sparse fields
        sa_cache_seqlens_buf = self._graph_sa_cache_seqlens[:bs]
        sa_cu_seqlens_k_buf = self._graph_sa_cu_seqlens_k[: bs + 1]
        sa_cu_seqlens_q_buf = self._graph_sa_cu_seqlens_q[: bs + 1]
        sa_cache_seqlens = compute_nsa_seqlens(cache_seqlens_buf, nsa_index_topk=self.sa_topk)
        sa_cache_seqlens_buf.copy_(sa_cache_seqlens)
        sa_cu_seqlens_k_buf[1:].copy_(
            torch.cumsum(sa_cache_seqlens, dim=0, dtype=torch.int32)
        )

        paged_mqa_schedule_metadata = None
        try:
            import deep_gemm

            paged_mqa_schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                cache_seqlens_buf, 64, deep_gemm.get_num_sms(),
            )
        except (ImportError, ModuleNotFoundError):
            paged_mqa_schedule_metadata = None

        metadata = KeyeSAAttnMetadata(
            page_size=self.page_size,
            cache_seqlens_int32=cache_seqlens_buf,
            max_seq_len_q=1,
            max_seq_len_k=real_page_table_buf.shape[1] * self.page_size,
            cu_seqlens_q=cu_seqlens_q_buf,
            cu_seqlens_k=cu_seqlens_k_buf,
            real_page_table=real_page_table_buf,
            page_table_1=self._graph_page_table_1[:bs],
            sa_cache_seqlens_int32=sa_cache_seqlens_buf,
            sa_cu_seqlens_q=sa_cu_seqlens_q_buf,
            sa_cu_seqlens_k=sa_cu_seqlens_k_buf,
            sa_seqlens_expanded=cache_seqlens_buf,
            sa_extend_seq_lens_list=[1] * bs,
            sa_max_seqlen_q=1,
            paged_mqa_schedule_metadata=paged_mqa_schedule_metadata,
        )
        self.decode_cuda_graph_metadata[bs] = metadata
        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info=None,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        """Update pre-allocated buffers in-place for CUDA graph replay."""
        assert forward_mode.is_decode_or_idle()
        assert seq_lens_cpu is not None

        metadata: KeyeSAAttnMetadata = self.decode_cuda_graph_metadata[bs]

        seq_lens_bs = seq_lens[:bs]
        seq_lens_cpu_bs = seq_lens_cpu[:bs]
        req_pool_indices_bs = req_pool_indices[:bs]

        max_len = int(seq_lens_cpu_bs.max().item())
        max_num_pages = self._graph_real_page_table.shape[1]

        # Update real_page_table (in-place)
        metadata.cache_seqlens_int32.copy_(seq_lens_bs.to(torch.int32))
        metadata.cu_seqlens_k[1 : bs + 1].copy_(
            torch.cumsum(seq_lens_bs.to(torch.int32), dim=0, dtype=torch.int32)
        )
        num_pages = (max_len + self.page_size - 1) // self.page_size
        strided = torch.arange(0, max_len, self.page_size, device=self.device, dtype=torch.int64)
        metadata.real_page_table[:bs, :num_pages].copy_(
            self.req_to_token[req_pool_indices_bs, :max_len][:, strided] // self.page_size
        )
        if num_pages < max_num_pages:
            metadata.real_page_table[:bs, num_pages:].zero_()

        # Update page_table_1 (token-level, in-place)
        metadata.page_table_1[:bs, :max_len].copy_(
            self.req_to_token[req_pool_indices_bs, :max_len].to(torch.int32)
        )
        if max_len < metadata.page_table_1.shape[1]:
            metadata.page_table_1[:bs, max_len:].zero_()

        # --- Update sparse metadata (in-place) ---
        sa_cache_seqlens = compute_nsa_seqlens(
            metadata.cache_seqlens_int32, nsa_index_topk=self.sa_topk
        )
        metadata.sa_cache_seqlens_int32.copy_(sa_cache_seqlens)
        metadata.sa_cu_seqlens_k[1 : bs + 1].copy_(
            torch.cumsum(sa_cache_seqlens, dim=0, dtype=torch.int32)
        )

        try:
            import deep_gemm

            new_schedule = deep_gemm.get_paged_mqa_logits_metadata(
                metadata.cache_seqlens_int32, 64, deep_gemm.get_num_sms()
            )
            if metadata.paged_mqa_schedule_metadata is None:
                metadata.paged_mqa_schedule_metadata = new_schedule
            else:
                metadata.paged_mqa_schedule_metadata.copy_(new_schedule)
        except (ImportError, ModuleNotFoundError):
            metadata.paged_mqa_schedule_metadata = None

        self.forward_metadata = metadata

    # ------------------------------------------------------------------
    # Attention forward
    # ------------------------------------------------------------------

    def forward(self, q, k, v, layer: "RadixAttention", forward_batch: ForwardBatch,
                save_kv_cache: bool = True, **kwargs):
        if forward_batch.forward_mode.is_extend_without_speculative():
            return self.forward_extend(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
        else:
            return self.forward_decode(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Prefill forward using sparse_attention_forward (effective_kernels)."""
        topk_indices = kwargs["topk_indices"]
        if k is not None and v is not None and save_kv_cache:
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        metadata = self.forward_metadata
        q = q.view(-1, self.num_q_heads, self.head_dim)

        assert metadata.sa_extend_seq_lens_list is not None
        unique_indices, q_mask, block_counts = topk_block_unique(
            topk_indices,
            metadata.cu_seqlens_q,
            topk_block=128 // self.num_kv_groups,
            max_seq_len=metadata.max_seq_len_k,
            max_model_len=self.max_model_len,
            page_table=metadata.page_table_1,
            is_sorted=self.deterministic_topk,
        )
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        output = sparse_attention_forward(
            q, k_cache, v_cache,
            metadata.cu_seqlens_q,
            unique_indices,
            q_mask,
            block_counts,
            self.sa_topk,
        )
        return output.reshape(-1, self.num_q_heads * self.head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Decode forward using flash_attn_with_kvcache with sparse page table."""
        assert _has_flash_attn, "flash_attn is required for decode"
        topk_indices = kwargs["topk_indices"]

        if k is not None and v is not None and save_kv_cache:
            cache_loc = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        metadata = self.forward_metadata
        q = q.view(-1, self.num_q_heads, self.head_dim)

        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        # Sparse path: use token-level page_table_1 (page_size=1), mirroring NSA _forward_fa3.
        # kv_cache is viewed with page_size=1 so each page_table entry is a physical slot id.
        token_slots = topk_indices
        if not self.deterministic_topk:
            token_slots = transform_index_page_table_decode(
                page_table=metadata.page_table_1,
                topk_indices=topk_indices,
                page_size=1,
            )
        k_cache_view = k_cache.view(-1, 1, self.num_kv_heads, self.head_dim)
        v_cache_view = v_cache.view(-1, 1, self.num_kv_heads, self.head_dim)

        result = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache_view,
            v_cache=v_cache_view,
            page_table=token_slots,
            cache_seqlens=metadata.sa_cache_seqlens_int32,
            cu_seqlens_q=metadata.sa_cu_seqlens_q,
            cu_seqlens_k_new=metadata.sa_cu_seqlens_k,
            max_seqlen_q=metadata.sa_max_seqlen_q,
            softmax_scale=layer.scaling,
            pack_gqa=True,
            softcap=layer.logit_cap,
            causal=True,
            return_softmax_lse=False,
        )
        return result.reshape(-1, self.num_q_heads * self.head_dim)
