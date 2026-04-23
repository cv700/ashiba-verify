"""Measure batched-Freivalds overhead across a size sweep, emit JSONL.

This is the JSONL-backed companion to benchmark_overhead.py. Same measurement
methodology as compare_naive_vs_batched.py (median of 10 iterations, device
sync inside each measurement), so the overhead numbers are directly comparable
to the `batched_overhead_pct` column in naive_vs_batched JSONL traces.

Usage:
    # M5 local (free, fast)
    python examples/benchmark_overhead_sweep.py \\
        --device mps \\
        --sizes 1024,2048,4096,8192,16384 \\
        > traces/2026-04-22_overhead_sweep_m5.jsonl

    # MI300X / H100 (remote)
    python examples/benchmark_overhead_sweep.py \\
        --device cuda \\
        --sizes 1024,2048,4096,8192,16384,32768 \\
        --silicon "AMD Instinct MI300X (gfx942)" \\
        --provider "DigitalOcean AMD Developer Cloud ATL1" \\
        > traces/2026-04-22_overhead_sweep_mi300x.jsonl
"""

import argparse
import json
import time
from typing import Optional

import torch


def freivalds_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    k: int = 10,
    tolerance: Optional[dict] = None,
) -> bool:
    """Batched Freivalds: 3 matmul calls against a (p, k) random matrix."""
    if tolerance is None:
        tolerance = {"atol": 1e-4, "rtol": 1e-4}

    m, _ = A.shape
    _, p = B.shape

    R = torch.empty(p, k, device=A.device, dtype=A.dtype).uniform_(-1.0, 1.0)
    BR = B @ R
    ABR = A @ BR
    CR = C @ R

    diff = (ABR - CR).abs()
    residual_per_iter = diff.amax(dim=0)
    cr_max_per_iter = CR.abs().amax(dim=0)
    threshold_per_iter = tolerance["atol"] + tolerance["rtol"] * cr_max_per_iter
    return not bool((residual_per_iter > threshold_per_iter).any().item())


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def median_time_ms(fn, n_iters=10):
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[n_iters // 2] * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sizes", type=str, default="1024,2048,4096,8192,16384,32768")
    parser.add_argument("--silicon", type=str, default="",
                        help="Descriptive silicon string for the trace header")
    parser.add_argument("--provider", type=str, default="",
                        help="Cloud provider / environment for the trace header")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    sizes = [int(s) for s in args.sizes.split(",")]

    header = {
        "event": "run_start",
        "device": str(device),
        "dtype": args.dtype,
        "k": args.k,
        "sizes": sizes,
        "pytorch": torch.__version__,
    }
    if args.silicon:
        header["silicon"] = args.silicon
    if args.provider:
        header["provider"] = args.provider
    if device.type == "cuda":
        header["cuda"] = torch.version.cuda or ""
        try:
            header["device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    print(json.dumps(header), flush=True)

    for n in sizes:
        try:
            A = torch.randn(n, n, device=device, dtype=dtype)
            B = torch.randn(n, n, device=device, dtype=dtype)
            C = A @ B
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(json.dumps({
                "event": "skip",
                "n": n,
                "reason": f"allocation failed: {type(e).__name__}",
            }), flush=True)
            continue

        # Warmup both
        for _ in range(3):
            _ = A @ B
            _ = freivalds_batched(A, B, C, k=args.k)
        sync(device)

        def matmul_fn():
            out = A @ B
            sync(device)

        def batched_fn():
            _ = freivalds_batched(A, B, C, k=args.k)
            sync(device)

        matmul_ms = median_time_ms(matmul_fn, n_iters=10)
        batched_ms = median_time_ms(batched_fn, n_iters=10)
        overhead_pct = 100.0 * batched_ms / matmul_ms

        print(json.dumps({
            "event": "measurement",
            "n": n,
            "matmul_ms": round(matmul_ms, 4),
            "freivalds_ms": round(batched_ms, 4),
            "overhead_pct": round(overhead_pct, 3),
            "k": args.k,
            "dtype": args.dtype,
        }), flush=True)

        del A, B, C
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(json.dumps({"event": "run_end"}), flush=True)


if __name__ == "__main__":
    main()
