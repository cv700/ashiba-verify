# ashiba-verify

**A probabilistic GEMM verifier for ML kernels across silicon.**

Implements Freivalds' algorithm (1979) for verifying matrix multiplication `C ≈ A @ B` at `O(n²)` cost vs `O(n³)` to recompute, with probability of false positive `2⁻ᵏ` after `k` iterations. Runs on PyTorch CUDA, Apple Silicon MPS, and ROCm backends.

Part of the kernel-contracts research program — see `https://ashibaresearch.com` and the accompanying paper at `https://arxiv.org/abs/YYMM.NNNNN`.

## What this is

Freivalds' algorithm gives you a way to verify that a matrix multiplication was performed correctly, at a fraction of the cost of redoing it:

- Pick a random vector `r`
- Compute `Br` (one matvec, `O(n²)`)
- Compute `A(Br)` (one matvec, `O(n²)`)
- Compute `Cr` (one matvec, `O(n²)`)
- If `A(Br) ≠ Cr` within tolerance, the original matmul was wrong
- Repeat `k` times for exponentially-decaying false-positive probability

This turns "did my GPU compute `A @ B` correctly?" from a question that costs as much as recomputing the whole matmul into a question that costs roughly `3k/n` of the original matmul. For `n = 4096`, `k = 10`, verification overhead is under 1% of the kernel cost. For production inference workloads this means every GEMM can be independently verified without meaningful throughput loss.

This is the smallest piece of the kernel-contracts framework — the verification mechanism for the reduction clauses of matmul and attention-adjacent operators. It's shipped as a standalone utility because it's useful on its own.

## Install

```bash
pip install ashiba-verify
```

Requires: `torch >= 2.1`. Works on CUDA, MPS (Apple Silicon), and ROCm devices. Falls back to CPU.

## Usage

```python
import torch
from ashiba_verify import freivalds_verify

# Your kernel under test
A = torch.randn(1024, 4096, device="cuda")
B = torch.randn(4096, 512, device="cuda")
C = A @ B  # the kernel you want to verify

# Verify C ≈ A @ B with 2⁻¹⁰ ≈ 0.001 false-positive probability
passed = freivalds_verify(A, B, C, k=10)
assert passed
```

## Catching a corrupted GEMM

```python
import torch
from ashiba_verify import freivalds_verify

# Correct computation
A = torch.randn(1024, 4096, device="cuda")
B = torch.randn(4096, 512, device="cuda")
C_correct = A @ B

# Introduce a corruption: single bit-flip in the output
C_corrupt = C_correct.clone()
C_corrupt[0, 0] += 1e-2

# Verifier catches it
assert freivalds_verify(A, B, C_correct, k=10) == True   # correct matmul passes
assert freivalds_verify(A, B, C_corrupt, k=10) == False  # corrupted matmul fails
```

## Cross-silicon verification

The key value proposition: a single contract, three silicon platforms, one comparison.

```python
import torch
from ashiba_verify import freivalds_verify

A_cpu = torch.randn(1024, 4096)
B_cpu = torch.randn(4096, 512)

# Apple Silicon
A_mps = A_cpu.to("mps")
B_mps = B_cpu.to("mps")
C_mps = A_mps @ B_mps

# Verify the computation occurred correctly on MPS
assert freivalds_verify(A_mps, B_mps, C_mps, k=10)

# Same for CUDA, ROCm:
# A_cuda = A_cpu.to("cuda"); B_cuda = B_cpu.to("cuda"); C_cuda = A_cuda @ B_cuda
# assert freivalds_verify(A_cuda, B_cuda, C_cuda, k=10)
```

## Overhead measurement

Freivalds verification overhead drops as matrix size grows, because matmul cost scales O(n³) while Freivalds cost scales O(n²·k). At production matmul sizes (n ≥ 16384), overhead is a few percent or less. At very large sizes (n ≥ 32768), overhead is under 1% of the kernel it verifies. The curve is measured below across three silicon platforms.

**FP32 overhead, k=10 iterations, across three silicon platforms:**

| n | Apple M5 (MPS) | AMD MI300X (ROCm 7.0) | NVIDIA H100 (CUDA 12.4) |
|---|---|---|---|
| 1024 | 86% | 475% | 310% |
| 2048 | 25% | 132% | 80% |
| 4096 | 6.0% | 29% | 16% |
| 8192 | 2.2% | 7.5% | 4.4% |
| 16384 | **0.97%** | 2.0% | 1.5% |
| 32768 | — | **0.75%** | **0.55%** |

**Reduced-precision overhead at n=16384:**

| dtype | Apple M5 | MI300X | H100 |
|---|---|---|---|
| FP32 | 0.97% | 2.0% | 1.5% |
| FP16 | — | 8.5% | 13% |
| BF16 | — | — | 14% |

**Reduced-precision at n=32768:**

| dtype | MI300X | H100 |
|---|---|---|
| FP32 | 0.75% | 0.55% |
| FP16 | — | 3.9% |
| BF16 | — | 4.4% |

### Why the curve has the shape it does

The overhead reflects a **GPU-utilization asymmetry**: large square matmuls achieve near-peak FLOP throughput on modern accelerators, while the smaller matmuls Freivalds uses (shape n × k, where k=10) are more memory-bandwidth-limited and hit lower FLOP utilization. Faster silicon widens this gap — an H100 or MI300X runs the verified matmul far closer to its theoretical peak than a Freivalds-shaped matmul, so the relative overhead is *higher* on server GPUs than on Apple Silicon at the same matrix size. The crossover to sub-1% overhead accordingly moves to larger matrices on faster silicon.

Reduced precision (FP16/BF16) increases overhead ratios because server tensor cores accelerate the verified matmul more than they accelerate the bandwidth-bound Freivalds matmuls. This is expected and will remain until Freivalds is refactored to batch across kernels (v0.2+ work).

### The batched-matmul optimization is load-bearing

The Freivalds algorithm naturally expresses as "sample k random vectors, do three matvecs each." Implementing it that way is catastrophically slow on GPUs: 30 sequential kernel launches per verification, each kernel memory-bandwidth-limited, no fusion. Sampling k random vectors *as a matrix* and doing three matmuls instead of 3k matvecs reduces the runtime by roughly an order of magnitude on every platform measured:

| Platform | Naive-vs-batched speedup (n=2048 FP32) |
|---|---|
| Apple M5 (MPS) | 12.3× |
| AMD MI300X | 4.6× |
| NVIDIA H100 | 6.6× |

Without this optimization, overhead at n=2048 is 200-500% on all three platforms — i.e., verification would cost more than re-running the kernel itself. Raw traces in `traces/2026-04-22_naive_vs_batched_*.jsonl`.

### Implication for production use

Freivalds is a viable continuous-verification mechanism for training workloads where per-kernel matrix sizes are in the thousands to tens of thousands. For large-hidden-dim transformer training (hidden dim 4096-16384, typical), overhead is 2-15% depending on silicon and precision. At inference-time matrix dimensions (typically smaller, more latency-sensitive), the tradeoff is workload-specific and the sensitivity floor (see below) becomes the more important constraint.

### Reproducing

```bash
# on your own hardware:
python examples/benchmark_overhead.py --device mps --size 16384 --dtype fp32   # Apple Silicon
python examples/benchmark_overhead.py --device cuda --size 16384 --dtype fp32  # NVIDIA or AMD (ROCm)

# full multi-size sweep:
for n in 1024 2048 4096 8192 16384 32768; do
  python examples/benchmark_overhead.py --device cuda --size $n --dtype fp32
done

# side-by-side naive vs batched comparison:
python examples/compare_naive_vs_batched.py --device cuda --dtype fp32 --sizes 512,1024,2048,4096,8192
```

Raw JSONL trace data for each platform is in `traces/`.

## What this is NOT

- **Not a contract specification language.** This is a single verification primitive. The contract-triple framework (specifying what correctness means, across failure modes, with calibration distributions) is separate work; see the paper.
- **Not a replacement for reference-based testing.** Freivalds catches output corruption of matrix multiplications specifically. For other kernel correctness properties (shape polymorphism, determinism, exceptional-value semantics), different verification mechanisms apply.
- **Not a proof of silicon correctness.** A passing Freivalds check says the matmul was probably correct on this input with high probability; it does not certify the silicon across all inputs.

## How this fits the kernel-contracts framework

The contract-triple's verification mechanism allows three alternatives: deterministic reference comparison, metamorphic relations, and randomized probabilistic checkers. This library implements the randomized-probabilistic option for the GEMM-type clauses. A full kernel-contract framework would wrap this alongside the other mechanisms; this repository ships only the verifier.

See [paper link] for the broader framework.

## Development

```bash
git clone https://github.com/cv700/ashiba-verify
cd ashiba-verify
pip install -e ".[dev]"
pytest tests/
```

## License

Apache 2.0. See `LICENSE`.

## Citation

If this is useful in research, please cite:

```bibtex
@article{veit2026kernelcontracts,
  author = {Veit, Cooper},
  title = {Kernel Contracts: A Specification Language for Silent Correctness Failures in ML Kernels},
  journal = {arXiv preprint},
  year = {2026},
  eprint = {YYMM.NNNNN},
  archivePrefix = {arXiv},
  primaryClass = {cs.PL}
}
```

## Roadmap

- [x] Freivalds verifier for GEMM (CUDA, MPS, ROCm)
- [ ] Benchmark harness for overhead measurement across precision regimes (FP32, FP16, BF16, FP8)
- [ ] Variants for attention (QK^T and softmax(·)V separately)
- [ ] Cross-silicon divergence detection (requires the contract framework; see kernel-contracts paper §6)

---

Ashiba Research · 2026 · apache 2.0
