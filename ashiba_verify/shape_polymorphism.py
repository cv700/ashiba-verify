"""C-CMP-03: Shape-Polymorphism Preservation.

Verify that a kernel claiming support over a shape class actually satisfies
its companion correctness contract over the full class, not merely at
benchmarked shapes.

A shape class C declares which input shapes the kernel claims to support
(e.g., "all M, N, K multiples of 16"). The verifier runs the kernel and
a reference implementation across both benchmarked and held-out shapes
inside C, then asks: does the held-out pass rate match the benchmarked
pass rate? If the kernel passes at benchmarked shapes but fails at
held-out shapes within the same declared class, the polymorphism
contract is violated.

Empirical context. Lange et al. (2025) document LLM-generated kernels
that pass at tested shapes and fail at untested shapes within the
claimed class. The Sakana CUDA Engineer post-mortem describes a kernel
that returned correct outputs only at the benchmarked GEMM shape and
silently produced incorrect outputs at every other shape in the
declared class. C-CMP-03 makes this failure mode nameable.

Usage. This module is a point-of-use test protocol. Invoke once against
a candidate kernel and a reference to obtain a single-shot pass/fail
verdict. It is not a continuous probe.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Contract and result types
# ---------------------------------------------------------------------------


@dataclass
class ShapeClass:
    """Declares a class of shapes the kernel claims to support.

    Attributes:
        name: Human-readable identifier
            (e.g., "GEMM-multiples-of-16").
        constraint: Predicate (shape_tuple) -> bool indicating membership.
        sample_shapes: Callable (n, seed) -> list of n shape-tuples drawn
            from the class.
    """

    name: str
    constraint: Callable[[Tuple[int, ...]], bool]
    sample_shapes: Callable[[int, Optional[int]], List[Tuple[int, ...]]]

    def contains(self, shape: Tuple[int, ...]) -> bool:
        return bool(self.constraint(shape))


@dataclass
class ShapeContract:
    """Declares a kernel's shape-polymorphism contract.

    Attributes:
        operation: Name of the operation (e.g., "matmul", "softmax").
        shape_class: The class of shapes the kernel claims to support.
        companion_tolerance: Dict with 'atol' and 'rtol' for the companion
            correctness check (kernel output vs. reference output).
    """

    operation: str
    shape_class: ShapeClass
    companion_tolerance: dict


@dataclass
class ShapePolymorphismResult:
    """Outcome of a single shape-polymorphism verification run."""

    contract: ShapeContract
    benchmarked_shapes: List[Tuple[int, ...]]
    held_out_shapes: List[Tuple[int, ...]]
    benchmarked_pass_rate: float
    held_out_pass_rate: float
    failed_held_out_shapes: List[Tuple[int, ...]]
    polymorphism_preserved: bool
    notes: str


# ---------------------------------------------------------------------------
# Verification logic
# ---------------------------------------------------------------------------


def verify_shape_polymorphism(
    kernel: Callable,
    reference: Callable,
    contract: ShapeContract,
    benchmarked_shapes: List[Tuple[int, ...]],
    n_held_out: int = 50,
    make_inputs: Optional[Callable] = None,
    seed: Optional[int] = None,
    polymorphism_threshold: float = 0.95,
) -> ShapePolymorphismResult:
    """Test whether a kernel honors its declared shape-polymorphism contract.

    For each benchmarked shape and each held-out shape, builds random
    inputs of that shape, runs `kernel` and `reference` on them, and
    checks if the outputs match within `contract.companion_tolerance`.
    Computes pass rates over both shape sets. The polymorphism property
    holds iff the held-out pass rate is at least `polymorphism_threshold`
    times the benchmarked pass rate.

    Args:
        kernel: Callable (*inputs) -> output. The kernel under test.
        reference: Callable (*inputs) -> output. Ground-truth implementation.
        contract: Shape contract.
        benchmarked_shapes: Shapes the kernel was tuned or tested on.
        n_held_out: Number of held-out shapes to test.
        make_inputs: Callable (shape, seed) -> tuple of input tensors.
            Defaults to a matmul-style 2-tensor builder.
        seed: Reproducibility seed.
        polymorphism_threshold: Held-out pass rate must be at least this
            fraction of benchmarked pass rate to count as preserved.
            Default 0.95.

    Returns:
        ShapePolymorphismResult.
    """
    if make_inputs is None:
        make_inputs = make_matmul_inputs

    # 1. Validate that benchmarked shapes are in the declared class
    for shape in benchmarked_shapes:
        if not contract.shape_class.contains(shape):
            raise ValueError(
                f"Benchmarked shape {shape} is not in declared shape class "
                f"{contract.shape_class.name!r}"
            )

    # 2. Sample held-out shapes from the class, excluding benchmarked
    benchmarked_set = set(benchmarked_shapes)
    candidates = contract.shape_class.sample_shapes(
        n_held_out * 3, seed=seed
    )
    held_out_shapes = [s for s in candidates if s not in benchmarked_set][
        :n_held_out
    ]
    # Retry once with a different seed if we still don't have enough
    if len(held_out_shapes) < n_held_out:
        retry_seed = (seed + 1) if seed is not None else None
        more = contract.shape_class.sample_shapes(
            n_held_out * 5, seed=retry_seed
        )
        for s in more:
            if s not in benchmarked_set and s not in held_out_shapes:
                held_out_shapes.append(s)
                if len(held_out_shapes) >= n_held_out:
                    break

    # 3. Test each shape: does kernel(inputs) match reference(inputs)?
    def shape_passes(shape: Tuple[int, ...]) -> bool:
        try:
            inputs = make_inputs(shape, seed=seed)
            k_out = kernel(*inputs)
            r_out = reference(*inputs)
            return bool(
                torch.allclose(k_out, r_out, **contract.companion_tolerance)
            )
        except Exception:
            return False

    benchmarked_results = [(s, shape_passes(s)) for s in benchmarked_shapes]
    held_out_results = [(s, shape_passes(s)) for s in held_out_shapes]

    bench_pass_rate = (
        sum(p for _, p in benchmarked_results) / len(benchmarked_results)
        if benchmarked_results
        else 0.0
    )
    held_out_pass_rate = (
        sum(p for _, p in held_out_results) / len(held_out_results)
        if held_out_results
        else 0.0
    )
    failed_held_out = [s for s, p in held_out_results if not p]

    # 4. Polymorphism property
    if bench_pass_rate == 0.0:
        polymorphism_preserved = held_out_pass_rate == 0.0
        notes = (
            "Kernel fails at benchmarked shapes; cannot meaningfully assess "
            "polymorphism. Verify the companion correctness contract first."
        )
    else:
        polymorphism_preserved = (
            held_out_pass_rate >= bench_pass_rate * polymorphism_threshold
        )
        if not polymorphism_preserved:
            n_failed = len(failed_held_out)
            notes = (
                f"Kernel passes {bench_pass_rate:.0%} of benchmarked shapes "
                f"but only {held_out_pass_rate:.0%} of held-out shapes "
                f"({n_failed} failures across {len(held_out_shapes)} "
                f"held-out). Compare: Lange et al. on LLM-generated kernels "
                "passing at tested shapes only; Sakana CUDA Engineer "
                "post-mortem."
            )
        else:
            notes = "Polymorphism preserved within threshold."

    return ShapePolymorphismResult(
        contract=contract,
        benchmarked_shapes=benchmarked_shapes,
        held_out_shapes=held_out_shapes,
        benchmarked_pass_rate=bench_pass_rate,
        held_out_pass_rate=held_out_pass_rate,
        failed_held_out_shapes=failed_held_out,
        polymorphism_preserved=polymorphism_preserved,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Default input builder for matmul
# ---------------------------------------------------------------------------


def make_matmul_inputs(
    shape: Tuple[int, int, int],
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build random inputs for a matmul of shape (M, N, K).

    Returns (A, B) where A is (M, K) and B is (K, N). When seeded, the
    same shape produces the same tensors across runs.
    """
    if seed is not None:
        torch.manual_seed(seed)
    M, N, K = shape
    return torch.randn(M, K), torch.randn(K, N)


# ---------------------------------------------------------------------------
# Standard shape classes
# ---------------------------------------------------------------------------


def _sample_gemm_mult16(
    n: int, seed: Optional[int] = None
) -> List[Tuple[int, int, int]]:
    """Sample GEMM shapes where M, N, K are all positive multiples of 16."""
    if seed is not None:
        torch.manual_seed(seed)
    multipliers = torch.randint(1, 64, (n, 3))
    return [
        (16 * int(a), 16 * int(b), 16 * int(c))
        for a, b, c in multipliers.tolist()
    ]


SHAPE_CLASS_GEMM_MULT16 = ShapeClass(
    name="GEMM-multiples-of-16",
    constraint=lambda s: (
        len(s) == 3 and all(d % 16 == 0 and d >= 16 for d in s)
    ),
    sample_shapes=_sample_gemm_mult16,
)


def _sample_gemm_any(
    n: int, seed: Optional[int] = None
) -> List[Tuple[int, int, int]]:
    """Sample arbitrary GEMM shapes with side length in [1, 1024]."""
    if seed is not None:
        torch.manual_seed(seed)
    return [
        (int(a), int(b), int(c))
        for a, b, c in torch.randint(1, 1024, (n, 3)).tolist()
    ]


SHAPE_CLASS_GEMM_ANY = ShapeClass(
    name="GEMM-any",
    constraint=lambda s: len(s) == 3 and all(d >= 1 for d in s),
    sample_shapes=_sample_gemm_any,
)


# ---------------------------------------------------------------------------
# Three-state calibration kernels (matmul)
# ---------------------------------------------------------------------------


def baseline_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Baseline state: returns zeros at the right output shape.

    Used to confirm the verifier detects total functional failure.
    """
    M, _ = A.shape
    _, N = B.shape
    return torch.zeros(M, N)


def bad_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    benchmarked_shape: Tuple[int, int, int] = (1024, 1024, 1024),
) -> torch.Tensor:
    """Bad state: hardcoded for one benchmarked shape.

    Returns the correct matmul at the benchmarked shape; returns zeros
    (silent failure) at every other shape. Sakana CUDA Engineer pattern.
    """
    M, K = A.shape
    K2, N = B.shape
    if (M, N, K) == benchmarked_shape:
        return torch.matmul(A, B)
    return torch.zeros(M, N)


def conforming_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Conforming state: shape-polymorphic by construction.

    Reference implementation for a contract with shape class GEMM-any.
    """
    return torch.matmul(A, B)


def reference_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Ground-truth reference for matmul polymorphism tests."""
    return torch.matmul(A, B)
