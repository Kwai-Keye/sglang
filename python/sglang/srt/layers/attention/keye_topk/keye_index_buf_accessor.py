"""
Keye FP8 Paged Index Buffer Accessors

Parameterized version of NSA's index_buf_accessor, supporting any head_dim in {32, 64, 128}.
The NSA originals hard-code head_dim=128; this file removes that restriction so Keye models
with head_dim=64 (or 32) can use the same FP8 paged K-cache format.

Buffer layout per page (identical to NSA, K-first + scale-last):
    page_size * head_dim  bytes  →  fp8_e4m3fn  K data   (all tokens' K data first)
    page_size * 4         bytes  →  float32     scale    (all tokens' scale after)

Total bytes per page: page_size * (head_dim + 4)

quant_block_size = min(head_dim, 128)
  - head_dim=128 → 1 block/token,  scale shape: [N, 1]
  - head_dim=64  → 1 block/token,  scale shape: [N, 1]  (block_size=64)
  - head_dim=32  → 1 block/token,  scale shape: [N, 1]  (block_size=32)

In all cases each token has exactly ONE scale value, so the per-page bytes for scale are
always page_size * 4 (one fp32 per token).

This layout is compatible with deep_gemm.fp8_paged_mqa_logits which internally extracts K
and scale from the fused buffer via torch.from_blob with stride kv_cache_stride_bytes:
    kv_cache        = from_blob(buf,              [pages, page_size, head_dim],
                                strides=[page_size*(head_dim+4), head_dim, 1])
    kv_cache_scales = from_blob(buf + K_region,   [pages, page_size],
                                strides=[page_size*(head_dim+4)/4, 1])

Write-address convention
------------------------
``loc`` must be the physical write address for each token:

    loc[i] = page_table[seq_pos[i] // page_size] * page_size + (seq_pos[i] % page_size)

This is exactly ``forward_batch.out_cache_loc`` — the same value NSATokenToKVPool uses
directly. KeyeIndexer.forward_cuda passes out_cache_loc without any remapping.
This ensures write and read use identical physical addresses.
"""

from typing import TYPE_CHECKING, Tuple

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KeyeTokenToKVPool


# ---------------------------------------------------------------------------
# SetKAndS — write FP8 K + float32 scale into paged buffer (K-first+scale-last)
# ---------------------------------------------------------------------------

def set_k_and_s(
    buf: torch.Tensor,            # [num_pages, page_size*(head_dim+4)], uint8
    loc: torch.Tensor,            # [N], int64 — write address (page=loc//page_size, off=loc%page_size)
    index_k: torch.Tensor,        # [N, head_dim], float8_e4m3fn
    index_k_scale: torch.Tensor,  # [N] or [N,1], float32
    page_size: int,
) -> None:
    """
    Write FP8 K and float32 scale into the paged buffer using K-first + scale-last layout.

    Buffer layout per page (same as NSA):
        bytes [0 .. page_size*head_dim)         → fp8 K data (all tokens)
        bytes [page_size*head_dim .. end)        → float32 scale (all tokens, 4B each)

    This matches deep_gemm.fp8_paged_mqa_logits which derives K and scale via from_blob
    with stride-based views over the fused buffer.

    ``loc[i]`` is the pre-computed write address for token i:
        loc[i] = page_table[seq_pos[i] // page_size] * page_size + (seq_pos[i] % page_size)
    """
    num_pages, buf_numel_per_page = buf.shape
    (num_tokens,) = loc.shape
    num_tokens_, index_head_dim = index_k.shape

    if index_k_scale.ndim == 1:
        num_tokens__ = index_k_scale.shape[0]
        scale_dim = 1
    elif index_k_scale.ndim == 2:
        num_tokens__, scale_dim = index_k_scale.shape
    else:
        raise ValueError(
            f"index_k_scale must be 1D or 2D, got shape {index_k_scale.shape}"
        )

    assert index_head_dim in (32, 64, 128), (
        f"keye_index_buf_accessor: index_head_dim={index_head_dim} must be 32, 64, or 128"
    )
    assert page_size == 64, f"page_size={page_size} must be 64"
    assert buf_numel_per_page == page_size * (index_head_dim + 4), (
        f"buf_numel_per_page={buf_numel_per_page} != page_size*({index_head_dim}+4)="
        f"{page_size*(index_head_dim+4)}"
    )
    assert num_tokens == num_tokens_ == num_tokens__, (
        f"Shape mismatch: loc={num_tokens}, k={num_tokens_}, scale={num_tokens__}"
    )
    assert scale_dim == 1

    assert buf.dtype == torch.uint8
    assert loc.dtype == torch.int64, f"loc.dtype={loc.dtype}, must be int64"
    assert index_k.dtype == torch.float8_e4m3fn
    assert index_k_scale.dtype == torch.float32

    assert buf.is_contiguous()
    assert loc.is_contiguous()
    assert index_k.is_contiguous()
    assert index_k_scale.is_contiguous()

    buf_fp8 = buf.view(torch.float8_e4m3fn)
    buf_fp32 = buf.view(torch.float32)

    _set_k_and_s_kernel[(num_tokens,)](
        buf_fp8,
        buf_fp32,
        loc,
        index_k,
        index_k_scale,
        index_k.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        NUM_K_ELEMS_PER_TOKEN=index_head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * index_head_dim,
    )


@triton.jit
def _set_k_and_s_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    loc_ptr,
    index_k_ptr,
    index_k_scale_ptr,
    index_k_ptr_stride_0,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    NUM_K_ELEMS_PER_TOKEN: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
):
    """One program per token to write.

    K-first + scale-last layout per page:
        K region:     bytes [0 .. PAGE_SIZE*NUM_K_ELEMS)
        scale region: bytes [PAGE_SIZE*NUM_K_ELEMS .. PAGE_SIZE*(NUM_K_ELEMS+4))

    ``loc`` encodes the physical write address:
        physical_page  = loc // PAGE_SIZE
        intra_page_off = loc % PAGE_SIZE

    The caller must ensure loc = page_table[seq_pos // PAGE_SIZE] * PAGE_SIZE + (seq_pos % PAGE_SIZE)
    so that this matches what the decode read kernel (fp8_paged_mqa_logits) expects.
    """
    token_id = tl.program_id(0)

    loc = tl.load(loc_ptr + token_id)

    in_k_offsets = token_id * index_k_ptr_stride_0 + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    k = tl.load(index_k_ptr + in_k_offsets)
    k_scale = tl.load(index_k_scale_ptr + token_id)

    loc_page_index = loc // PAGE_SIZE
    loc_token_offset_in_page = loc % PAGE_SIZE

    out_k_offsets = (
        loc_page_index * BUF_NUMEL_PER_PAGE
        + loc_token_offset_in_page * NUM_K_ELEMS_PER_TOKEN
        + tl.arange(0, NUM_K_ELEMS_PER_TOKEN)
    )

    # "//4" because fp32 pointer steps are 4-byte aligned
    out_s_offset = (
        loc_page_index * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + loc_token_offset_in_page
    )

    tl.store(buf_fp8_ptr + out_k_offsets, k)
    tl.store(buf_fp32_ptr + out_s_offset, k_scale)


# ---------------------------------------------------------------------------
# GetKAndS — read FP8 K + float32 scale from paged buffer (K-first+scale-last)
# ---------------------------------------------------------------------------

def get_k_and_s(
    buf: torch.Tensor,           # [num_pages, page_size*(head_dim+4)], uint8
    page_indices: torch.Tensor,  # [num_pages], int32/int64
    seq_len: int,
    page_size: int,
    index_head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused gather of FP8 K and float32 scale from paged buffer (K-first + scale-last layout).
    Returns:
        k_fp8:   [seq_len, index_head_dim]  uint8  (fp8_e4m3fn bytes)
        k_scale: [seq_len, 4]               uint8  (float32 bytes; reinterpret-cast as needed)
    """
    s_offset_in_page = page_size * index_head_dim

    k_out = torch.empty((seq_len, index_head_dim), dtype=torch.uint8, device=buf.device)
    s_out = torch.empty((seq_len, 4), dtype=torch.uint8, device=buf.device)

    grid = (seq_len,)
    _get_k_and_s_kernel[grid](
        buf,
        page_indices,
        k_out,
        s_out,
        seq_len,
        page_size,
        buf.shape[1],          # buf_numel_per_page
        index_head_dim,
        s_offset_in_page,
        BLOCK_SIZE_K=index_head_dim,
    )

    return k_out, s_out


@triton.jit
def _get_k_and_s_kernel(
    buf_ptr,
    page_indices_ptr,
    k_out_ptr,
    s_out_ptr,
    seq_len: tl.constexpr,
    page_size: tl.constexpr,
    buf_numel_per_page: tl.constexpr,
    index_head_dim: tl.constexpr,
    s_offset_in_page: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused kernel: one program per token, gathers K + S in a single pass.

    K-first + scale-last layout:
        K src:     page_base + token_off * head_dim
        scale src: page_base + s_offset_in_page + token_off * 4
    """
    token_id = tl.program_id(0)

    page_idx = token_id // page_size
    token_offset_in_page = token_id % page_size

    page_index = tl.load(page_indices_ptr + page_idx)

    # ----- K data -----
    k_src_base = page_index * buf_numel_per_page + token_offset_in_page * index_head_dim
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_offsets < index_head_dim
    k_data = tl.load(buf_ptr + k_src_base + k_offsets, mask=k_mask)
    tl.store(k_out_ptr + token_id * index_head_dim + k_offsets, k_data, mask=k_mask)

    # ----- S (scale) data -----
    s_src_base = page_index * buf_numel_per_page + s_offset_in_page + token_offset_in_page * 4
    s_offsets = tl.arange(0, 4)
    s_data = tl.load(buf_ptr + s_src_base + s_offsets)
    tl.store(s_out_ptr + token_id * 4 + s_offsets, s_data)


# ---------------------------------------------------------------------------
# Pool-level wrapper classes (same interface as NSA index_buf_accessor)
# ---------------------------------------------------------------------------

class SetKAndS:
    @classmethod
    def execute(
        cls,
        pool: "KeyeTokenToKVPool",
        buf: torch.Tensor,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        set_k_and_s(
            buf=buf,
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
            page_size=pool.page_size,
        )


class GetKAndS:
    @classmethod
    def execute(
        cls,
        pool: "KeyeTokenToKVPool",
        buf: torch.Tensor,
        seq_len: int,
        page_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return get_k_and_s(
            buf=buf,
            page_indices=page_indices,
            seq_len=seq_len,
            page_size=pool.page_size,
            index_head_dim=pool.index_head_dim,
        )
