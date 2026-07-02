#!/usr/bin/env python3
"""Throughput checks for FA2-with-4D-mask using Triton benchmark helpers."""

from __future__ import annotations

import argparse
import gc
import sys
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from triton.testing import Benchmark, do_bench, do_bench_cudagraph, perf_report

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.precision_check import CASES, Case, fa2_call, make_inputs, parse_dtype


def sdpa_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    score: Optional[torch.Tensor],
    *,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    attn_mask = None
    if score is not None:
        attn_mask = score
    if mask is not None:
        additive = torch.zeros(mask.shape, device=mask.device, dtype=torch.float32)
        additive.masked_fill_(~mask, float("-inf"))
        attn_mask = additive if attn_mask is None else attn_mask + additive
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=True,
        )


def sdpa_default(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    score: Optional[torch.Tensor],
    *,
    scale: float,
    is_causal: bool,
) -> torch.Tensor:
    attn_mask = score
    if mask is not None:
        if attn_mask is None:
            attn_mask = mask
        else:
            additive = torch.zeros(mask.shape, device=mask.device, dtype=torch.float32)
            additive.masked_fill_(~mask, float("-inf"))
            attn_mask = attn_mask + additive
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=True,
    )


def estimate_tflops(B: int, Tq: int, Tk: int, Hq: int, D: int, is_causal: bool, ms: float, mode: str) -> float:
    causal_factor = 0.5 if is_causal else 1.0
    # fwd has qk and pv matmuls. The fwd_bwd callable times forward plus
    # autograd backward, whose FlashAttention backward recomputes qk and then
    # forms dP, dV, dQ, and dK. Elementwise softmax work is not counted.
    multiplier = 4.0 if mode == "fwd" else 14.0
    flops = multiplier * causal_factor * B * Hq * Tq * Tk * D
    return flops * 1e-12 / (ms * 1e-3)


def provider_parts(provider: str) -> tuple[str, str]:
    if provider.endswith("_fwd_bwd"):
        return provider[: -len("_fwd_bwd")], "fwd_bwd"
    if provider.endswith("_fwd"):
        return provider[: -len("_fwd")], "fwd"
    raise ValueError(provider)


def run_one_case(args: argparse.Namespace, case: Case) -> None:
    variable_vals = args.Tq_values
    memory_records = []
    B, Hq, Hk, D = args.B, args.Hq, args.Hk, args.D
    provider_vals = [f"{backend}_{mode}" for mode in args.bench_modes for backend in args.providers]
    benchmark_args = {
        "B": B,
        "Tk": args.Tk,
        "Hq": Hq,
        "Hk": Hk,
        "D": D,
        "dtype": args.dtype,
        "device": args.device,
        "case": case,
        "rep": args.rep,
        "warmup": args.warmup,
        "cudagraph": args.cudagraph,
        "mask_keep_prob": args.mask_keep_prob,
        "score_scale": args.score_scale,
        "seed": args.seed,
        "use_atomic_backward": args.use_atomic_backward,
        "measure_memory": args.measure_memory,
    }

    @perf_report(
        Benchmark(
            x_names=["Tq"],
            x_vals=variable_vals,
            x_log=len(variable_vals) > 1,
            line_arg="provider",
            line_vals=provider_vals,
            line_names=provider_vals,
            styles=[("blue", "-"), ("green", "-"), ("blue", "--"), ("green", "--")][: len(provider_vals)],
            ylabel="TFLOPS",
            plot_name=f"fa2_{case.name}",
            args=benchmark_args,
        )
    )
    def benchmark(Tq: int, provider: str, **bench_kwargs):
        backend, mode = provider_parts(provider)
        torch.manual_seed(bench_kwargs["seed"])
        local_args = argparse.Namespace(
            B=bench_kwargs["B"],
            Tq=Tq,
            Tk=bench_kwargs["Tk"] if bench_kwargs["Tk"] > 0 else Tq,
            Hq=bench_kwargs["Hq"],
            Hk=bench_kwargs["Hk"],
            D=bench_kwargs["D"],
            dtype=bench_kwargs["dtype"],
            device=bench_kwargs["device"],
            mask_keep_prob=bench_kwargs["mask_keep_prob"],
            score_scale=bench_kwargs["score_scale"],
        )
        q = k = v = mask = score = grad = None
        try:
            q, k, v, mask, score = make_inputs(local_args, bench_kwargs["case"])
        except Exception as exc:
            print(f"[WARN] input allocation failed for provider={provider} case={bench_kwargs['case'].name} Tq={Tq}: {exc}")
            torch.cuda.empty_cache()
            gc.collect()
            return 0.0, 0.0, 0.0
        scale = D ** -0.5
        if backend == "tilelang":
            call = partial(
                fa2_call,
                q,
                k,
                v,
                mask,
                score,
                scale=scale,
                is_causal=bench_kwargs["case"].is_causal,
                use_atomic_backward=bench_kwargs["use_atomic_backward"],
            )
        elif backend == "sdpa":
            call = partial(sdpa_flash, q, k, v, mask, score, scale=scale, is_causal=bench_kwargs["case"].is_causal)
        elif backend == "sdpa_default":
            call = partial(sdpa_default, q, k, v, mask, score, scale=scale, is_causal=bench_kwargs["case"].is_causal)
        else:
            raise ValueError(provider)

        if mode == "fwd_bwd":
            q = q.detach().requires_grad_(True)
            k = k.detach().requires_grad_(True)
            v = v.detach().requires_grad_(True)
            grad = torch.randn((local_args.B, local_args.Tq, local_args.Hq, local_args.D), device=q.device, dtype=q.dtype).transpose(1, 2)
            if backend == "tilelang":
                call = partial(
                    fa2_call,
                    q,
                    k,
                    v,
                    mask,
                    score,
                    scale=scale,
                    is_causal=bench_kwargs["case"].is_causal,
                    use_atomic_backward=bench_kwargs["use_atomic_backward"],
                )
            elif backend == "sdpa":
                call = partial(sdpa_flash, q, k, v, mask, score, scale=scale, is_causal=bench_kwargs["case"].is_causal)
            else:
                call = partial(sdpa_default, q, k, v, mask, score, scale=scale, is_causal=bench_kwargs["case"].is_causal)

            def fn():
                q.grad = None
                k.grad = None
                v.grad = None
                out = call()
                out.backward(grad)
        else:
            def fn():
                return call()

        quantiles = [0.5, 0.2, 0.8]
        try:
            if bench_kwargs["measure_memory"]:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(torch.device(bench_kwargs["device"]))
                baseline = torch.cuda.memory_allocated(torch.device(bench_kwargs["device"]))
                if mode == "fwd":
                    with torch.no_grad():
                        out = fn()
                    del out
                else:
                    fn()
                    q.grad = None
                    k.grad = None
                    v.grad = None
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated(torch.device(bench_kwargs["device"]))
                memory_records.append(
                    (
                        int(Tq),
                        provider,
                        baseline / 1024**2,
                        peak / 1024**2,
                        (peak - baseline) / 1024**2,
                    )
                )
                torch.cuda.empty_cache()

            if mode == "fwd":
                with torch.no_grad():
                    fn()
                    torch.cuda.synchronize()
                    if bench_kwargs["cudagraph"]:
                        ms, min_ms, max_ms = do_bench_cudagraph(fn, rep=bench_kwargs["rep"], quantiles=quantiles)
                    else:
                        ms, min_ms, max_ms = do_bench(fn, warmup=bench_kwargs["warmup"], rep=bench_kwargs["rep"], quantiles=quantiles)
            else:
                fn()
                torch.cuda.synchronize()
                ms, min_ms, max_ms = do_bench(fn, warmup=bench_kwargs["warmup"], rep=bench_kwargs["rep"], quantiles=quantiles)
        except Exception as exc:
            print(f"[WARN] {provider} failed for case={bench_kwargs['case'].name} Tq={Tq}: {exc}")
            return 0.0, 0.0, 0.0
        finally:
            del q, k, v, mask, score, grad
            torch.cuda.empty_cache()
            gc.collect()

        return (
            estimate_tflops(B, Tq, local_args.Tk, Hq, D, bench_kwargs["case"].is_causal, ms, mode),
            estimate_tflops(B, Tq, local_args.Tk, Hq, D, bench_kwargs["case"].is_causal, min_ms, mode),
            estimate_tflops(B, Tq, local_args.Tk, Hq, D, bench_kwargs["case"].is_causal, max_ms, mode),
        )

    benchmark.run(show_plots=False, print_data=True)
    if memory_records:
        print(f"{case.name}_memory_mib:")
        print(f"{'Tq':>8}  {'provider':>22}  {'baseline':>12}  {'peak':>12}  {'extra':>12}")
        for Tq, provider, baseline, peak, extra in memory_records:
            print(f"{Tq:8d}  {provider:>22}  {baseline:12.1f}  {peak:12.1f}  {extra:12.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", type=parse_dtype, default=torch.float16)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--Tq-values", type=int, nargs="+", default=[8192])
    parser.add_argument("--Tk", type=int, default=8192, help="Use <=0 to follow Tq.")
    parser.add_argument("--Hq", type=int, default=32)
    parser.add_argument("--Hk", type=int, default=8)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES), default=["no_causal", "causal"])
    parser.add_argument("--providers", nargs="+", choices=["tilelang", "sdpa", "sdpa_default"], default=["tilelang", "sdpa"])
    parser.add_argument("--bench-modes", nargs="+", choices=["fwd", "fwd_bwd"], default=["fwd"])
    parser.add_argument("--use-atomic-backward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-keep-prob", type=float, default=0.75)
    parser.add_argument("--score-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--cudagraph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--measure-memory", action=argparse.BooleanOptionalAction, default=False, help="Print per-provider CUDA allocated-memory peak/extra MiB for one measured run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if torch.device(args.device).type == "cuda":
        torch.cuda.set_device(torch.device(args.device))
    print("torch", torch.__version__)
    print("device", torch.cuda.get_device_name(torch.device(args.device)) if torch.cuda.is_available() else args.device)
    for case_name in args.cases:
        print(f"\n=== throughput {case_name} ===")
        run_one_case(args, CASES[case_name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
