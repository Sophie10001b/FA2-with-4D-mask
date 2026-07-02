#!/usr/bin/env python3
"""Precision checks for FA2-with-4D-mask.

The reference path is intentionally PyTorch eager math, not SDPA, so mask and
additive score semantics are easy to inspect. Inputs use BHSD layout by default:
Q [B, Hq, Tq, D], K/V [B, Hk, Tk, D].
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.fa2 import flash_attention_with_4d_mask


@dataclass(frozen=True)
class Case:
    name: str
    is_causal: bool = False
    use_mask: bool = False
    use_score: bool = False


CASES = {
    "no_causal": Case("no_causal"),
    "causal": Case("causal", is_causal=True),
    "random_mask": Case("random_mask", use_mask=True),
    "score": Case("score", use_score=True),
}


def parse_dtype(name: str) -> torch.dtype:
    table = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {name}") from exc


def fa2_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    score: Optional[torch.Tensor],
    *,
    scale: float,
    is_causal: bool,
    use_atomic_backward: bool = True,
    bhsd_input: bool = True,
) -> torch.Tensor:
    # autograd.Function.apply does not accept kwargs, so keep this positional.
    return flash_attention_with_4d_mask(
        q,
        k,
        v,
        mask,
        score,
        scale,
        is_causal,
        use_atomic_backward,
        bhsd_input,
    )


def sdpa_flash_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=True,
        )


def make_mask(B: int, Hq: int, Tq: int, Tk: int, keep_prob: float, device: torch.device) -> torch.Tensor:
    mask = torch.empty((B, Hq, Tq, Tk), device=device, dtype=torch.bool)
    mask.bernoulli_(keep_prob)
    # Guarantee at least one valid key per query row to avoid all -inf softmax rows.
    q_idx = torch.arange(Tq, device=device)
    safe_k = torch.clamp(q_idx, max=Tk - 1)
    mask[:, :, q_idx, safe_k] = True
    return mask.contiguous()


def make_inputs(args: argparse.Namespace, case: Case) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    device = torch.device(args.device)
    dtype = args.dtype
    q = torch.randn((args.B, args.Tq, args.Hq, args.D), device=device, dtype=dtype)
    k = torch.randn((args.B, args.Tk, args.Hk, args.D), device=device, dtype=dtype)
    v = torch.randn((args.B, args.Tk, args.Hk, args.D), device=device, dtype=dtype)
    mask = make_mask(args.B, args.Hq, args.Tq, args.Tk, args.mask_keep_prob, device) if case.use_mask else None
    score = None
    if case.use_score:
        score = (torch.randn((args.B, args.Hq, args.Tq, args.Tk), device=device, dtype=torch.float32) * args.score_scale).contiguous()
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask, score


def eager_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    score: Optional[torch.Tensor],
    *,
    scale: float,
    is_causal: bool,
    ref_block: int = 0,
) -> torch.Tensor:
    B, Hq, Tq, D = q.shape
    _, Hk, Tk, _ = k.shape
    assert Hq % Hk == 0, f"Hq={Hq} must be divisible by Hk={Hk}"
    groups = Hq // Hk
    if groups != 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

    qf = q.float()
    kf = k.float()
    vf = v.float()

    def chunk_ref(q_start: int, q_end: int) -> torch.Tensor:
        q_chunk = qf[:, :, q_start:q_end, :]
        logits = torch.einsum("bhqd,bhkd->bhqk", q_chunk, kf)
        if score is not None:
            logits = logits + score[:, :, q_start:q_end, :].float()
        logits = logits * scale

        valid = None
        if is_causal:
            q_idx = torch.arange(q_start, q_end, device=q.device)[:, None]
            k_idx = torch.arange(Tk, device=q.device)[None, :]
            valid = (q_idx >= k_idx).view(1, 1, q_end - q_start, Tk)
        if mask is not None:
            mask_chunk = mask[:, :, q_start:q_end, :]
            valid = mask_chunk if valid is None else (valid & mask_chunk)
        if valid is not None:
            logits = logits.masked_fill(~valid, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        return torch.einsum("bhqk,bhkd->bhqd", probs, vf)

    if ref_block and ref_block > 0 and ref_block < Tq:
        chunks = [chunk_ref(q_start, min(q_start + ref_block, Tq)) for q_start in range(0, Tq, ref_block)]
        return torch.cat(chunks, dim=2).to(q.dtype)
    return chunk_ref(0, Tq).to(q.dtype)


def diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    af = actual.float()
    ef = expected.float()
    diff = (af - ef).abs()
    denom = ef.abs().clamp_min(1e-6)
    rel = diff / denom
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": rel.max().item(),
        "mean_rel": rel.mean().item(),
    }


def print_stats(prefix: str, stats: dict[str, float]) -> None:
    print(
        f"{prefix}: max_abs={stats['max_abs']:.6g} "
        f"mean_abs={stats['mean_abs']:.6g} max_rel={stats['max_rel']:.6g} "
        f"mean_rel={stats['mean_rel']:.6g}"
    )


def case_passed(stats: dict[str, float], atol: float, rtol: float) -> bool:
    return stats["max_abs"] <= atol or stats["mean_rel"] <= rtol


def run_forward_case(args: argparse.Namespace, case: Case) -> bool:
    print(f"\n=== forward {case.name} ===")
    torch.manual_seed(args.seed)
    q, k, v, mask, score = make_inputs(args, case)
    scale = args.scale if args.scale is not None else args.D ** -0.5
    print(
        f"shape: Q={tuple(q.shape)} K={tuple(k.shape)} V={tuple(v.shape)} "
        f"mask={None if mask is None else tuple(mask.shape)} "
        f"score={None if score is None else tuple(score.shape)} causal={case.is_causal}"
    )

    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        actual = fa2_call(q, k, v, mask, score, scale=scale, is_causal=case.is_causal)
        torch.cuda.synchronize()
        print(f"fa2 forward ok in {(time.perf_counter() - t0) * 1e3:.3f} ms")
    except Exception:
        print("fa2 forward FAILED")
        traceback.print_exc(limit=args.traceback_limit)
        return False

    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        expected = eager_ref(q, k, v, mask, score, scale=scale, is_causal=case.is_causal, ref_block=args.ref_block)
        torch.cuda.synchronize()
        print(f"eager ref ok in {(time.perf_counter() - t0) * 1e3:.3f} ms")
    except Exception:
        print("eager ref FAILED")
        traceback.print_exc(limit=args.traceback_limit)
        return False

    stats = diff_stats(actual, expected)
    print_stats("forward diff", stats)
    ok = case_passed(stats, args.atol, args.rtol)

    sdpa_out = None
    if args.check_sdpa and mask is None and score is None:
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            sdpa_out = sdpa_flash_ref(q, k, v, scale=scale, is_causal=case.is_causal)
            torch.cuda.synchronize()
            print(f"sdpa flash ref ok in {(time.perf_counter() - t0) * 1e3:.3f} ms")
            sdpa_eager_stats = diff_stats(sdpa_out, expected)
            print_stats("sdpa vs eager diff", sdpa_eager_stats)
            fa2_sdpa_stats = diff_stats(actual, sdpa_out)
            print_stats("fa2 vs sdpa diff", fa2_sdpa_stats)
            ok = case_passed(sdpa_eager_stats, args.atol, args.rtol) and ok
            ok = case_passed(fa2_sdpa_stats, args.atol, args.rtol) and ok
        except Exception:
            print("sdpa flash ref FAILED")
            traceback.print_exc(limit=args.traceback_limit)
            ok = False

    print("forward result:", "PASS" if ok else "FAIL")
    del q, k, v, mask, score, actual, expected, sdpa_out
    torch.cuda.empty_cache()
    gc.collect()
    return ok


def run_backward_case(args: argparse.Namespace, case: Case) -> bool:
    print(f"\n=== backward {case.name} ===")
    torch.manual_seed(args.seed)
    q, k, v, mask, score = make_inputs(args, case)
    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)
    if mask is not None:
        mask = mask.detach()
    if score is not None:
        score = score.detach()
    scale = args.scale if args.scale is not None else args.D ** -0.5
    grad = torch.randn((args.B, args.Tq, args.Hq, args.D), device=q.device, dtype=q.dtype).transpose(1, 2)

    try:
        actual = fa2_call(q, k, v, mask, score, scale=scale, is_causal=case.is_causal, use_atomic_backward=args.use_atomic_backward)
        actual.backward(grad)
        torch.cuda.synchronize()
        grads_actual = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())
        print("fa2 backward ok")
    except Exception:
        print("fa2 backward FAILED")
        traceback.print_exc(limit=args.traceback_limit)
        return False

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    try:
        expected = eager_ref(q_ref, k_ref, v_ref, mask, score, scale=scale, is_causal=case.is_causal, ref_block=args.ref_block)
        expected.backward(grad)
        torch.cuda.synchronize()
        grads_expected = (q_ref.grad, k_ref.grad, v_ref.grad)
        print("eager backward ok")
    except Exception:
        print("eager backward FAILED")
        traceback.print_exc(limit=args.traceback_limit)
        return False

    ok = True
    for name, actual_grad, expected_grad in zip(("dQ", "dK", "dV"), grads_actual, grads_expected):
        stats = diff_stats(actual_grad, expected_grad)
        print_stats(f"{name} diff", stats)
        ok = case_passed(stats, args.atol, args.rtol) and ok
    print("backward result:", "PASS" if ok else "FAIL")
    torch.cuda.empty_cache()
    gc.collect()
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", type=parse_dtype, default=torch.float16)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--Tq", type=int, default=8192)
    parser.add_argument("--Tk", type=int, default=8192)
    parser.add_argument("--Hq", type=int, default=32)
    parser.add_argument("--Hk", type=int, default=8)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-keep-prob", type=float, default=0.75)
    parser.add_argument("--score-scale", type=float, default=0.1)
    parser.add_argument("--ref-block", type=int, default=128, help="Query rows per eager-reference block; <=0 uses a full logits tensor.")
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES), default=list(CASES))
    parser.add_argument("--check-backward", action="store_true")
    parser.add_argument("--check-sdpa", action=argparse.BooleanOptionalAction, default=True, help="Also compare no-mask/no-score cases with PyTorch SDPA flash backend.")
    parser.add_argument("--use-atomic-backward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--traceback-limit", type=int, default=30)
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 after printing failures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_device(torch.device(args.device))
    print("torch", torch.__version__)
    print("device", torch.cuda.get_device_name(torch.device(args.device)) if torch.cuda.is_available() else args.device)
    print("dtype", args.dtype)

    ok = True
    for case_name in args.cases:
        case = CASES[case_name]
        ok = run_forward_case(args, case) and ok
        if args.check_backward:
            ok = run_backward_case(args, case) and ok
    return 0 if (ok or args.no_fail) else 1


if __name__ == "__main__":
    raise SystemExit(main())
