"""Compare the naive (pre-fix) and batched (post-fix) Freivalds implementations.

The naive version does 3*k sequential matvecs. The batched version does 3 matmuls
against a (p, k) random matrix. Both have identical FLOP counts; the wall-time
difference is GPU-utilization asymmetry: matvecs are memory-bandwidth-limited and
don't saturate modern GPUs, while matmuls are FLOP-limited and do.

This script reconstructs both implementations for forensic comparison. Output is
JSON-Lines to stdout for easy parsing.

Usage:
    python examples/compare_naive_vs_batched.py --device mps
"""

import argparse
import json
import time
from typing import Optional

import torch


def freivalds_naive(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    k: int = 10,
    tolerance: Optional[dict] = None,
) -> bool:
    """Pre-fix Freivalds — per-iteration matvec loop.

    This was the original implementation that produced 240% overhead at n=2048.
    Preserved verbatim for comparison.
    """
    if tolerance is None:
        tolerance = {"atol": 1e-4, "rtol": 1e-4}

    m, n = A.shape
    _, p = B.shape
    passed = True

    for _ in range(k):
        r = torch.empty(p, device=A.device, dtype=A.dtype).uniform_(-1.0, 1.0)
        br = B @ r       # (n,)   — matvec #1
        abr = A @ br     # (m,)   — matvec #2
        cr = C @ r       # (m,)   — matvec #3

        diff = (abr - cr).abs()
        residual = diff.max().item()
        cr_max = cr.abs().max().item()
        threshold = tolerance["atol"] + tolerance["rtol"] * cr_max
        if residual > threshold:
            passed = False
    return passed


def freivalds_batched(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    k: int = 10,
    tolerance: Optional[dict] = None,
) -> bool:
    """Post-fix Freivalds — batched matmul.

    Same FLOP count as naive, but 3 matmuls instead of 3*k matvecs.
    """
    if tolerance is None:
        tolerance = {"atol": 1e-4, "rtol": 1e-4}

    m, n = A.shape
    _, p = B.shape

    R = torch.empty(p, k, device=A.device, dtype=A.dtype).uniform_(-1.0, 1.0)
    BR = B @ R           # (n, k) — matmul #1
    ABR = A @ BR         # (m, k) — matmul #2
    CR = C @ R           # (m, k) — matmul #3

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


def time_it(fn, n_warmup=3, n_iters=10):
    for _ in range(n_warmup):
        fn()
    sync_device = None
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = fn()
        # synchronize inside the measurement since we want wall time
        if hasattr(result, "device"):
            sync(result.device)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sizes", type=str, default="1024,2048,4096,8192",
                        help="Comma-separated list of matrix sizes")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    sizes = [int(s) for s in args.sizes.split(",")]

    # Header
    print(json.dumps({
        "event": "run_start",
        "device": str(device),
        "dtype": args.dtype,
        "k": args.k,
        "sizes": sizes,
    }))

    for n in sizes:
        A = torch.randn(n, n, device=device, dtype=dtype)
        B = torch.randn(n, n, device=device, dtype=dtype)
        C = A @ B

        # Warmup all three
        _ = A @ B
        _ = freivalds_naive(A, B, C, k=args.k)
        _ = freivalds_batched(A, B, C, k=args.k)
        sync(device)

        # Measure matmul
        def matmul_fn():
            out = A @ B
            sync(device)
            return out
        matmul_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            matmul_fn()
            matmul_times.append(time.perf_counter() - t0)
        matmul_ms = sorted(matmul_times)[5] * 1000

        # Measure naive
        def naive_fn():
            r = freivalds_naive(A, B, C, k=args.k)
            sync(device)
            return r
        naive_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            naive_fn()
            naive_times.append(time.perf_counter() - t0)
        naive_ms = sorted(naive_times)[5] * 1000

        # Measure batched
        def batched_fn():
            r = freivalds_batched(A, B, C, k=args.k)
            sync(device)
            return r
        batched_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            batched_fn()
            batched_times.append(time.perf_counter() - t0)
        batched_ms = sorted(batched_times)[5] * 1000

        speedup = naive_ms / batched_ms
        naive_overhead = 100.0 * naive_ms / matmul_ms
        batched_overhead = 100.0 * batched_ms / matmul_ms

        print(json.dumps({
            "event": "measurement",
            "n": n,
            "matmul_ms": round(matmul_ms, 3),
            "naive_ms": round(naive_ms, 3),
            "batched_ms": round(batched_ms, 3),
            "naive_overhead_pct": round(naive_overhead, 2),
            "batched_overhead_pct": round(batched_overhead, 2),
            "batched_speedup_vs_naive": round(speedup, 1),
        }))

    print(json.dumps({"event": "run_end"}))


if __name__ == "__main__":
    main()
