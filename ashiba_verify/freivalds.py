"""Freivalds' algorithm for probabilistic matrix-multiplication verification.

Freivalds (1979) "Probabilistic Machines Can Use Less Running Time."

Given an alleged product C = A @ B:
    - sample a random vector r
    - check that A(Br) == Cr within tolerance
    - if they disagree, the matmul was wrong
    - repeat k times for 2^{-k} false-positive probability

Cost per iteration: O(n^2) matvecs. Total: O(k*n^2) vs O(n^3) for recomputation.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class FreivaldsResult:
    """Result of a Freivalds verification.

    Attributes:
        passed: True iff all k iterations agreed within tolerance.
        k: Number of iterations performed.
        max_residual: Largest observed ||A(Br) - Cr||_inf across iterations.
        tolerance_used: Absolute + relative tolerance actually applied.
        false_positive_probability: 2^{-k}.
    """

    passed: bool
    k: int
    max_residual: float
    tolerance_used: dict
    false_positive_probability: float


def _default_tolerance(dtype: torch.dtype) -> dict:
    """Tolerance derived from the precision class of the matmul.

    Tolerance is specified as a relative bound against the infinity-norm of
    the Freivalds reference vector cr = C @ r. This is the principled choice
    for Freivalds verification because the numerical error in the check
    (|A(Br) - Cr|_inf) scales with matrix norms, not with pointwise output
    magnitudes — using pointwise relative tolerance fails near zero crossings
    of cr regardless of the matmul's actual correctness.

    The check is: max(|A(Br) - Cr|) <= atol + rtol * max(|Cr|).

    Defaults are conservative. Users can pass explicit tolerance to tighten.
    """
    if dtype == torch.float64:
        return {"atol": 1e-10, "rtol": 1e-10}
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    if dtype == torch.bfloat16:
        # BF16 has 8 mantissa bits; unit roundoff ~4e-3; conservative floor
        return {"atol": 1e-2, "rtol": 5e-2}
    if dtype == torch.float16:
        # FP16 accumulation on silicon with FP16 accumulators can reach ~2% relative
        return {"atol": 1e-2, "rtol": 5e-2}
    if getattr(torch, "float8_e4m3fn", None) is not None and dtype == torch.float8_e4m3fn:
        return {"atol": 5e-2, "rtol": 1e-1}
    # conservative fallback
    return {"atol": 1e-2, "rtol": 5e-2}


def freivalds_verify(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    k: int = 10,
    tolerance: Optional[dict] = None,
    seed: Optional[int] = None,
    return_result: bool = False,
):
    """Verify that C ≈ A @ B via Freivalds' algorithm.

    Args:
        A: (m, n) matrix.
        B: (n, p) matrix.
        C: (m, p) alleged product.
        k: Number of iterations. Each adds a factor of 1/2 to false-positive prob.
        tolerance: Dict with 'atol' and 'rtol'. If None, derived from A.dtype.
        seed: If set, makes the random vectors reproducible.
        return_result: If True, returns a FreivaldsResult; otherwise returns bool.

    Returns:
        bool (passed) or FreivaldsResult (if return_result=True).

    Raises:
        ValueError: If shapes are inconsistent.
        RuntimeError: If A, B, C are on different devices or have incompatible dtypes.
    """
    # Shape check
    if A.dim() != 2 or B.dim() != 2 or C.dim() != 2:
        raise ValueError(
            f"All inputs must be 2D matrices; got A.dim()={A.dim()}, "
            f"B.dim()={B.dim()}, C.dim()={C.dim()}"
        )
    m, n = A.shape
    n2, p = B.shape
    if n != n2:
        raise ValueError(
            f"Inner dimensions must match: A is ({m}, {n}), B is ({n2}, {p})"
        )
    if C.shape != (m, p):
        raise ValueError(
            f"Output shape mismatch: expected ({m}, {p}), got {tuple(C.shape)}"
        )

    # Device/dtype consistency
    if not (A.device == B.device == C.device):
        raise RuntimeError(
            f"All tensors must be on the same device; got {A.device}, {B.device}, {C.device}"
        )

    # Tolerance
    if tolerance is None:
        tolerance = _default_tolerance(A.dtype)

    # Batched sampling: sample all k random vectors as a (p, k) matrix so the
    # Freivalds iterations become three matmuls (B @ R, A @ (BR), C @ R) instead
    # of 3*k matvecs. This is the critical optimization for GPU performance —
    # matvecs are memory-bandwidth-limited and don't saturate GPU throughput,
    # while matmuls are FLOP-limited and achieve near-peak. The asymptotic FLOP
    # cost is identical; wall-time is dramatically lower.
    # MPS has quirks with torch.Generator on non-CPU devices; sample on CPU
    # when seeded and transfer, which is portable across all backends.
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        R_cpu = torch.empty(p, k, dtype=A.dtype).uniform_(-1.0, 1.0, generator=generator)
    else:
        R_cpu = torch.empty(p, k, dtype=A.dtype).uniform_(-1.0, 1.0)
    R = R_cpu.to(A.device) if A.device.type != "cpu" else R_cpu

    # Three batched matmuls: B @ R, A @ (BR), C @ R
    BR = B @ R          # (n, k)
    ABR = A @ BR        # (m, k)
    CR = C @ R          # (m, k)

    # Compare via per-iteration norm-based tolerance, then aggregate.
    # residual_per_iter[j] = max |ABR[:, j] - CR[:, j]|_inf
    # cr_max_per_iter[j]   = max |CR[:, j]|_inf
    # threshold[j]         = atol + rtol * cr_max_per_iter[j]
    # passes iff residual_per_iter[j] <= threshold[j] for all j in [0, k).
    diff = (ABR - CR).abs()
    residual_per_iter = diff.amax(dim=0)       # (k,)
    cr_max_per_iter = CR.abs().amax(dim=0)     # (k,)
    threshold_per_iter = tolerance["atol"] + tolerance["rtol"] * cr_max_per_iter
    fail_mask = residual_per_iter > threshold_per_iter

    max_residual = residual_per_iter.max().item()
    passed = not bool(fail_mask.any().item())

    if return_result:
        return FreivaldsResult(
            passed=passed,
            k=k,
            max_residual=max_residual,
            tolerance_used=tolerance,
            false_positive_probability=2.0 ** (-k),
        )
    return passed
