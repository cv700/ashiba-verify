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

Freivalds verification overhead drops as matrix size grows, because matmul cost scales O(n³) while Freivalds verification cost scales O(n²·k). At production-scale matrix sizes typical of transformer training (n ≥ 8192), overhead is under a few percent. At very large sizes (n ≥ 16384) overhead crosses below 1% of the matmul it verifies.

**Measured on Apple M5 with MPS backend, FP32, k=10 Freivalds iterations:**

| Matrix size | Matmul time | Freivalds time | Overhead |
|---|---|---|---|
| 1024 × 1024 | 0.93 ms | 0.80 ms | 86% |
| 2048 × 2048 | 4.97 ms | 1.23 ms | 25% |
| 4096 × 4096 | 44.1 ms | 2.66 ms | 6.0% |
| 8192 × 8192 | 391 ms | 8.5 ms | 2.2% |
| **16384 × 16384** | **3610 ms** | **34.9 ms** | **0.97%** |

The overhead curve reflects a GPU-utilization asymmetry: large square matmuls achieve near-peak FLOP throughput, while the smaller matmuls Freivalds uses (shape n × k, where k is the iteration count) are more memory-bandwidth-limited. Batching k iterations into a single matmul (shape n × k instead of k separate matvecs) is the critical optimization — without it, overhead is 20-200× worse.

**Implication for production use:** Freivalds is a viable production-scale verification mechanism for training workloads where per-kernel matrix sizes are in the thousands. For inference workloads with smaller matrix dimensions, the overhead is proportionally higher and the tradeoff depends on the latency budget.

Reproducing:

```bash
python examples/benchmark_overhead.py --device mps --size 16384 --dtype fp32
python examples/benchmark_overhead.py --device cuda --size 16384 --dtype fp16  # (on CUDA hardware)
```

Cross-platform benchmarks (NVIDIA H100, AMD MI300X) pending; see `TRACES.md`.

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
