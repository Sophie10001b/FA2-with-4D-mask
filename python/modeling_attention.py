#!/usr/bin/env python3
"""User-facing wrappers for the TileLang FA2 4D-mask attention kernel.

The core kernel lives in :mod:`python.fa2`.  This module keeps the call surface
close to the attention functions commonly used from ``modeling_*.py`` files:
inputs are BHSD by default, optional 4D mask/score tensors are accepted, and a
HuggingFace-style ``(attn_output, attn_weights)`` adapter is provided.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

from .fa2 import flash_attention_with_4d_mask

Layout = Literal["bhsd", "bshd"]


def _shape_from_layout(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, layout: Layout) -> tuple[int, int, int, int, int]:
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query, key, and value must be rank-4 tensors")
    if layout == "bhsd":
        batch, heads_q, seqlen_q, head_dim = query.shape
        batch_k, heads_k, seqlen_k, head_dim_k = key.shape
        batch_v, heads_v, seqlen_v, head_dim_v = value.shape
    elif layout == "bshd":
        batch, seqlen_q, heads_q, head_dim = query.shape
        batch_k, seqlen_k, heads_k, head_dim_k = key.shape
        batch_v, seqlen_v, heads_v, head_dim_v = value.shape
    else:
        raise ValueError(f"unsupported layout: {layout}")

    if batch_k != batch or batch_v != batch:
        raise ValueError("query, key, and value must have the same batch size")
    if seqlen_k != seqlen_v:
        raise ValueError("key and value must have the same sequence length")
    if heads_k != heads_v:
        raise ValueError("key and value must have the same number of heads")
    if head_dim_k != head_dim or head_dim_v != head_dim:
        raise ValueError("query, key, and value must have the same head dimension")
    if heads_q % heads_k != 0:
        raise ValueError(f"query heads ({heads_q}) must be divisible by key/value heads ({heads_k})")
    return batch, heads_q, seqlen_q, seqlen_k, head_dim


def _expand_4d(x: torch.Tensor, *, batch: int, heads: int, seqlen_q: int, seqlen_k: int, name: str) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError(f"{name} must have shape [B, H or 1, Tq, Tk]")
    if x.shape[-2:] != (seqlen_q, seqlen_k):
        raise ValueError(f"{name} has trailing shape {tuple(x.shape[-2:])}, expected {(seqlen_q, seqlen_k)}")
    if x.shape[0] not in (1, batch):
        raise ValueError(f"{name} batch dimension must be 1 or {batch}")
    if x.shape[1] not in (1, heads):
        raise ValueError(f"{name} head dimension must be 1 or {heads}")
    if x.shape[0] != batch or x.shape[1] != heads:
        x = x.expand(batch, heads, seqlen_q, seqlen_k)
    return x.contiguous()


def _merge_masks(lhs: Optional[torch.Tensor], rhs: torch.Tensor) -> torch.Tensor:
    return rhs if lhs is None else (lhs & rhs)


def _merge_scores(lhs: Optional[torch.Tensor], rhs: torch.Tensor) -> torch.Tensor:
    return rhs if lhs is None else (lhs + rhs)


def tilelang_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    score: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    is_causal: bool = False,
    layout: Layout = "bhsd",
    attention_mask_is_scaled: bool = True,
    use_atomic_backward: bool = True,
) -> torch.Tensor:
    """Run TileLang FA2 attention.

    Args:
        query/key/value: Rank-4 tensors.  Default layout is BHSD
            ``[batch, heads, seqlen, head_dim]``.  Set ``layout="bshd"`` for
            ``[batch, seqlen, heads, head_dim]``.
        attention_mask: Optional 4D bool mask or additive attention bias with
            shape ``[B, Hq or 1, Tq, Tk]``.  Bool mask uses True for valid
            entries.  Float masks are treated as additive bias on scaled logits
            by default, matching PyTorch SDPA/HF convention.
        mask: Optional explicit bool 4D mask.  True means keep.
        score: Optional explicit float32 4D score added to ``QK`` before scale.
        scale: Softmax scale.  Defaults to ``head_dim ** -0.5``.
        is_causal: Use the kernel causal path without materializing a 4D mask.
        attention_mask_is_scaled: If True, float ``attention_mask`` is divided
            by ``scale`` before passing to the kernel, because the kernel score
            is added before scaling.
        use_atomic_backward: Use the atomic backward implementation.

    Returns:
        Attention output in the same layout as the input.
    """
    batch, heads_q, seqlen_q, seqlen_k, head_dim = _shape_from_layout(query, key, value, layout)
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"TileLang attention expects fp16 or bf16 inputs, got {query.dtype}")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("query, key, and value must have the same dtype")
    if scale is None:
        scale = head_dim ** -0.5
    scale = float(scale)

    if mask is not None:
        if mask.dtype != torch.bool:
            raise ValueError("mask must be a bool tensor with True for valid entries")
        mask = _expand_4d(mask, batch=batch, heads=heads_q, seqlen_q=seqlen_q, seqlen_k=seqlen_k, name="mask")

    if score is not None:
        if not torch.is_floating_point(score):
            raise ValueError("score must be a floating-point tensor")
        score = _expand_4d(score.to(device=query.device, dtype=torch.float32), batch=batch, heads=heads_q, seqlen_q=seqlen_q, seqlen_k=seqlen_k, name="score")

    if attention_mask is not None:
        attention_mask = _expand_4d(attention_mask.to(device=query.device), batch=batch, heads=heads_q, seqlen_q=seqlen_q, seqlen_k=seqlen_k, name="attention_mask")
        if attention_mask.dtype == torch.bool:
            mask = _merge_masks(mask, attention_mask)
        elif torch.is_floating_point(attention_mask):
            additive = attention_mask.to(dtype=torch.float32)
            if attention_mask_is_scaled:
                additive = additive / scale
            score = _merge_scores(score, additive)
        else:
            raise ValueError("attention_mask must be bool or floating point")

    return flash_attention_with_4d_mask(
        query,
        key,
        value,
        mask,
        score,
        scale,
        bool(is_causal),
        bool(use_atomic_backward),
        layout == "bhsd",
    )


def tilelang_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """HuggingFace-style attention function adapter.

    This has the same broad shape as the attention helper functions used by
    many ``modeling_*.py`` files: it receives BHSD query/key/value tensors and
    returns ``(attn_output, attn_weights)``.  Attention weights are not
    materialized, so the second return value is always ``None``.
    """
    if dropout not in (0, 0.0, None):
        raise ValueError("TileLang FA2 wrapper does not implement attention dropout")

    layout = kwargs.pop("layout", "bhsd")
    mask = kwargs.pop("mask", None)
    score = kwargs.pop("score", None)
    use_atomic_backward = kwargs.pop("use_atomic_backward", True)
    attention_mask_is_scaled = kwargs.pop("attention_mask_is_scaled", True)
    is_causal = kwargs.pop("is_causal", None)
    if is_causal is None:
        is_causal = attention_mask is None and bool(getattr(module, "is_causal", True))

    out = tilelang_flash_attention(
        query,
        key,
        value,
        attention_mask=attention_mask,
        mask=mask,
        score=score,
        scale=scaling,
        is_causal=bool(is_causal),
        layout=layout,
        attention_mask_is_scaled=attention_mask_is_scaled,
        use_atomic_backward=use_atomic_backward,
    )
    return out, None


__all__ = ["tilelang_flash_attention", "tilelang_attention_forward"]
