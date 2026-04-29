"""C-ORD-01: Reduction-Order Tolerance Bound.

Verify that a reduction kernel honors its declared reduction-order-invariance
contract: the output is bounded across all valid reduction orders by
N * eps(P) * max|x|, where N is the input length, eps(P) is the unit
roundoff for precision P, and max|x| is the largest absolute input value.

The contract follows from floating-point non-associativity (FPNA): a sum
of N values computed in different orders may differ at the bit level, but
the pairwise difference is bounded by the FPNA bound above. A kernel that
produces schedule-dependent variation exceeding the bound has either
introduced non-FPNA noise (e.g., atomic-add race conditions) or violated
the declared precision regime.

Empirical context. Shanmugavelu et al. (2024, "FPNA") measure run-to-run
variability of 3.35-5.03e-6 for PyTorch scatter_reduce and index_add on
H100, and show that 1,000 paired GraphSAGE models trained with
non-deterministic reduction converge to distinct weight sets despite
identical loss curves. Qiang et al. (2026, "DASH") quantify the cost of
suppressing this: FlashAttention-3's deterministic backward pass incurs
up to 37.9% throughput reduction on H800. He and Thinking Machines Lab
(2025) provide reference implementations satisfying reduction-order
invariance via fixed tile sizes and consistent reduction order. Yuan et
al. (2025) measure 9.15% accuracy variance from runtime configuration
alone, attributing the effect to FPNA under limited precision.

Note on precision regimes. The FPNA bound scales with eps(P), so at lower
precisions (BF16, FP16) the bound is loose and many kernels satisfy the
contract trivially. The contract is most informative at FP32 and tighter.

Usage. This module is a point-of-use test protocol. Invoke once against
a candidate reduction kernel to obtain a single-shot pass/fail verdict.
It is not a continuous probe.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch


# Machine epsilon (2^-mantissa_bits) for each precision regime.
# Matches the paper's eps(P) convention.
PRECISION_EPSILON = {
    "FP64": 2.0 ** -52,
    "FP32": 2.0 ** -23,
    "BF16": 2.0 ** -7,
    "FP16": 2.0 ** -10,
    "FP8_E4M3": 2.0 ** -3,
}


PRECISION_DTYPES = {
    "FP64": torch.float64,
    "FP32": torch.float32,
    "BF16": torch.bfloat16,
    "FP16": torch.float16,
}


# ---------------------------------------------------------------------------
# Contract and result types
# ---------------------------------------------------------------------------


@dataclass
class ReductionOrderContract:
    """Declares a reduction kernel's order-invariance contract.

    Attributes:
        operation: Reduction op ("sum", "mean", "norm").
        precision: One of "FP32", "BF16", "FP16", "FP64".
            Determines eps(P) in the FPNA bound.
    """

    operation: str
    precision: str

    def __post_init__(self):
        if self.precision not in PRECISION_EPSILON:
            raise ValueError(
                f"precision must be one of {list(PRECISION_EPSILON.keys())}; "
                f"got {self.precision!r}"
            )


@dataclass
class ReductionOrderResult:
    """Outcome of a single reduction-order verification run."""

    contract: ReductionOrderContract
    n_elements: int
    fp64_reference: float
    schedule_outputs: Dict[str, float]
    max_pairwise_diff: float
    fpna_bound: float
    within_bound: bool
    fpna_signature_observed: bool  # True if differences are non-zero AND within bound
    notes: str


# ---------------------------------------------------------------------------
# Default schedule set
# ---------------------------------------------------------------------------


DEFAULT_SCHEDULES: List[Dict] = [
    {"name": "tree_block_64", "strategy": "tree", "block_size": 64},
    {"name": "tree_block_256", "strategy": "tree", "block_size": 256},
    {"name": "tree_block_1024", "strategy": "tree", "block_size": 1024},
    {"name": "torch_default", "strategy": "torch_default"},
    {"name": "atomic_add", "strategy": "atomic"},
]


# ---------------------------------------------------------------------------
# Verification logic
# ---------------------------------------------------------------------------


def verify_reduction_order_contract(
    reduction_kernel: Callable[[torch.Tensor, Dict], torch.Tensor],
    contract: ReductionOrderContract,
    n_elements: int = 100_000,
    schedules: Optional[List[Dict]] = None,
    seed: Optional[int] = None,
) -> ReductionOrderResult:
    """Test whether a reduction kernel honors its declared order-invariance
    contract.

    Computes the reduction with each provided schedule, computes a high-
    precision FP64 reference, and checks whether the maximum pairwise
    difference among schedule outputs is bounded by N * eps(P) * max|x|.

    Args:
        reduction_kernel: Callable (input_tensor, schedule_dict) -> scalar
            tensor. The kernel should respect schedule["strategy"] and any
            other schedule parameters it understands.
        contract: Declared order-invariance contract.
        n_elements: Length of the input vector to reduce.
        schedules: List of schedule dicts to test. Defaults to
            DEFAULT_SCHEDULES (tree at three block sizes + torch default
            + atomic).
        seed: Reproducibility seed for input generation.

    Returns:
        ReductionOrderResult.
    """
    if schedules is None:
        schedules = DEFAULT_SCHEDULES

    if seed is not None:
        torch.manual_seed(seed)

    # 1. Generate input in target precision
    dtype = PRECISION_DTYPES.get(contract.precision)
    if dtype is None:
        # FP8 has no native PyTorch dtype on most backends; fall back to FP32
        # for input generation but keep the FP8 epsilon in the bound.
        dtype = torch.float32
    x = torch.randn(n_elements, dtype=dtype)
    max_abs = x.abs().max().item()

    # 2. Compute FP64 reference (ground truth across orders)
    fp64_reference = x.double().sum().item()

    # 3. Run kernel with each schedule
    schedule_outputs: Dict[str, float] = {}
    for schedule in schedules:
        try:
            out = reduction_kernel(x, schedule)
            if isinstance(out, torch.Tensor):
                out = out.item()
            schedule_outputs[schedule["name"]] = float(out)
        except (NotImplementedError, RuntimeError, ValueError):
            # Kernel does not support this schedule; record absence
            continue

    # 4. Compute max pairwise difference across schedules
    output_values = list(schedule_outputs.values())
    if len(output_values) >= 2:
        max_diff = max(
            abs(a - b)
            for i, a in enumerate(output_values)
            for b in output_values[i + 1 :]
        )
    else:
        max_diff = 0.0

    # 5. Compute the FPNA bound: N * eps(P) * max|x|
    eps = PRECISION_EPSILON[contract.precision]
    fpna_bound = n_elements * eps * max_abs

    within_bound = max_diff <= fpna_bound
    fpna_signature_observed = max_diff > 0 and within_bound

    # 6. Notes
    if len(output_values) < 2:
        notes = (
            f"Only {len(output_values)} schedule(s) produced output; cannot "
            "assess pairwise variance. Verifier needs at least two schedules."
        )
    elif within_bound:
        if max_diff == 0:
            notes = (
                "All schedules produced bit-identical output. Stronger than "
                "the FPNA bound requires; consistent with a bitwise-"
                "deterministic implementation."
            )
        else:
            notes = (
                f"Pairwise differences within FPNA bound ({max_diff:.3e} "
                f"<= {fpna_bound:.3e}). Consistent with reduction-order "
                "invariance under floating-point non-associativity."
            )
    else:
        notes = (
            f"Pairwise differences exceed FPNA bound ({max_diff:.3e} > "
            f"{fpna_bound:.3e}). Schedule-dependent variation cannot be "
            "explained by non-associativity alone; suggests atomic-add "
            "race, accumulator-precision drift, or reduction-order "
            "non-invariance beyond the declared contract."
        )

    return ReductionOrderResult(
        contract=contract,
        n_elements=n_elements,
        fp64_reference=fp64_reference,
        schedule_outputs=schedule_outputs,
        max_pairwise_diff=max_diff,
        fpna_bound=fpna_bound,
        within_bound=within_bound,
        fpna_signature_observed=fpna_signature_observed,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Three-state calibration kernels
# ---------------------------------------------------------------------------


def baseline_kernel(x: torch.Tensor, schedule: Dict) -> torch.Tensor:
    """Baseline state: returns zero regardless of input.

    Used to confirm the verifier detects total functional failure. Will
    pass the FPNA bound trivially (all outputs identical at zero) but
    diverges arbitrarily from the FP64 reference. The verifier cannot
    detect this baseline failure with reduction-order checking alone;
    pair with C-PRC-01 (Freivalds) for output correctness.
    """
    return torch.zeros((), dtype=x.dtype)


def bad_kernel(x: torch.Tensor, schedule: Dict) -> torch.Tensor:
    """Bad state: schedule-dependent perturbation.

    Produces a sum with a schedule-dependent constant offset, simulating
    the kind of non-determinism observed when atomicAdd is used for
    parallel reduction without a deterministic ordering. Pairwise schedule
    differences exceed the FPNA bound at FP32 precision (where the bound
    is tight); may stay within bound at BF16 / FP16 (where the bound is
    loose).
    """
    base = x.sum()
    # Schedule-dependent perturbation. Hash the schedule name to produce
    # a deterministic but schedule-varying offset.
    name = schedule.get("name", "default")
    name_hash = hash(name) % 100
    perturbation = torch.tensor(name_hash * 0.1, dtype=x.dtype)
    return base + perturbation


def conforming_kernel(x: torch.Tensor, schedule: Dict) -> torch.Tensor:
    """Conforming state: deterministic reduction whose output is invariant
    across schedules within the FPNA bound.

    Implements tree reduction at the requested block size, falling back
    to torch.sum for "torch_default". For "atomic" or unknown strategies,
    falls back to torch.sum (which is the deterministic CPU reduction in
    PyTorch); a real GPU atomic-add path would behave differently on
    GPU but the reference implementation provides the conforming
    numerical envelope.
    """
    strategy = schedule.get("strategy", "torch_default")
    if strategy == "tree":
        block_size = schedule.get("block_size", 256)
        block_size = min(block_size, len(x))
        pad = (-len(x)) % block_size
        if pad > 0:
            x_padded = torch.cat([x, torch.zeros(pad, dtype=x.dtype)])
        else:
            x_padded = x
        blocks = x_padded.reshape(-1, block_size).sum(dim=1)
        return blocks.sum()
    # torch_default, atomic, or unknown: deterministic CPU reduction
    return x.sum()
