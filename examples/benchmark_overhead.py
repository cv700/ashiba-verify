"""Benchmark: measure Freivalds verification overhead as fraction of matmul cost.

Usage:
    python examples/benchmark_overhead.py --device cuda --size 4096 --dtype fp16
    python examples/benchmark_overhead.py --device mps --size 2048 --dtype fp32

Expected overhead: under 1% of matmul cost at size >= 2048 with k=10 iterations.
"""

import argparse
import time

import torch

from ashiba_verify import freivalds_verify


DTYPES = {
    "fp64": torch.float64,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def bench_matmul(A, B, n_warmup=3, n_iters=20):
    """Return median matmul time in seconds."""
    # warmup
    for _ in range(n_warmup):
        _ = A @ B
    _sync(A.device)

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = A @ B
        _sync(A.device)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def bench_freivalds(A, B, C, k, n_warmup=3, n_iters=20):
    """Return median Freivalds verification time in seconds."""
    for _ in range(n_warmup):
        _ = freivalds_verify(A, B, C, k=k)
    _sync(A.device)

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = freivalds_verify(A, B, C, k=k)
        _sync(A.device)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def _sync(device):
    """Device synchronization for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to benchmark on.",
    )
    parser.add_argument("--size", type=int, default=4096, help="Matrix dimension (square matmul).")
    parser.add_argument("--dtype", choices=list(DTYPES.keys()), default="fp32")
    parser.add_argument("--k", type=int, default=10, help="Number of Freivalds iterations.")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    n = args.size

    print(f"Device:      {device}")
    print(f"Size:        {n} x {n} x {n}")
    print(f"Dtype:       {args.dtype}")
    print(f"k:           {args.k}")
    print(f"FP coverage: 1 - 2^-{args.k} = {1 - 2.0**(-args.k):.6f}")
    print()

    A = torch.randn(n, n, device=device, dtype=dtype)
    B = torch.randn(n, n, device=device, dtype=dtype)
    C = A @ B

    matmul_s = bench_matmul(A, B)
    freivalds_s = bench_freivalds(A, B, C, k=args.k)

    overhead_pct = 100.0 * freivalds_s / matmul_s

    print(f"matmul time:      {matmul_s * 1000:.3f} ms")
    print(f"freivalds time:   {freivalds_s * 1000:.3f} ms ({args.k} iterations)")
    print(f"overhead:         {overhead_pct:.2f}% of matmul cost")
    print()
    print("expected: overhead < 1% at n >= 2048")


if __name__ == "__main__":
    main()
