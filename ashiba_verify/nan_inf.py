"""C-EXC-01: NaN/Inf Propagation Semantics.

Verify that a kernel honors its declared NaN/Inf propagation policy.

A kernel claims one of three policies:
    IEEE_PROPAGATE - NaN/Inf flow per IEEE 754-2019 to all dependent outputs
    MASK           - NaN/Inf masked per documented mask (caller-supplied)
    RAISE          - kernel raises on encountering NaN or Inf in input

The verifier injects NaN or Inf at specified input positions, runs the
kernel once, and compares the actual output exceptional-value positions
to the expected positions per the declared policy.

Empirical context. Wen et al. (2025), "Mind the Gap," document multiple
silent IEEE 754 violations across silicon platforms:
  - aten::batch_norm on AMD replaces NaN with interpolated values from
    nearby data
  - aten::reshape on Apple Metal silently converts NaN to 0
  - aten::remainder and aten::convolution produce Inf on AMD and Intel
    where NVIDIA returns NaN
  - Positive-integer division by zero diverges three ways: NVIDIA returns
    2^32 - 1, Mac returns 0, AMD returns dividend + 1
Each is a silent, undocumented deviation from IEEE 754. C-EXC-01 makes
the deviation nameable.

Usage. This module is a point-of-use test protocol. Invoke once against
a candidate kernel to obtain a single-shot pass/fail verdict. It is not
a continuous probe.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


VALID_POLICIES = ("IEEE_PROPAGATE", "MASK", "RAISE")
VALID_INJECTION_TYPES = ("nan", "pos_inf", "neg_inf")


# ---------------------------------------------------------------------------
# Contract and result types
# ---------------------------------------------------------------------------


@dataclass
class NaNInfContract:
    """Declares the NaN/Inf propagation policy a kernel claims to honor.

    Attributes:
        operation: Name of the kernel being verified
            (e.g., "elementwise_add", "batch_norm", "softmax").
        declared_policy: One of "IEEE_PROPAGATE", "MASK", "RAISE".
        documented_mask: Required iff declared_policy == "MASK". A short
            string identifying the mask convention (e.g., "zero",
            "nearest", "saturate").
    """

    operation: str
    declared_policy: str
    documented_mask: Optional[str] = None

    def __post_init__(self):
        if self.declared_policy not in VALID_POLICIES:
            raise ValueError(
                f"declared_policy must be one of {VALID_POLICIES}; "
                f"got {self.declared_policy!r}"
            )
        if self.declared_policy == "MASK" and self.documented_mask is None:
            raise ValueError(
                "MASK policy requires documented_mask to be specified"
            )


@dataclass
class NaNPropagationResult:
    """Outcome of a single NaN/Inf verification run."""

    contract: NaNInfContract
    n_injections: int
    injection_type: str
    raised_exception: Optional[str]
    inputs_had_exceptional: bool
    output_has_exceptional: bool
    expected_exceptional_positions: Optional[torch.Tensor]
    actual_exceptional_positions: Optional[torch.Tensor]
    matches_declared_policy: bool
    silent_replacement_detected: Optional[str]
    notes: str


# ---------------------------------------------------------------------------
# Verification logic
# ---------------------------------------------------------------------------


def verify_nan_inf_contract(
    kernel: Callable[[torch.Tensor], torch.Tensor],
    contract: NaNInfContract,
    base_input_shape: Tuple[int, ...] = (1024,),
    n_injections: int = 10,
    injection_type: str = "nan",
    propagation_predictor: Optional[
        Callable[[torch.Tensor], torch.Tensor]
    ] = None,
    seed: Optional[int] = None,
) -> NaNPropagationResult:
    """Test whether a kernel honors its declared NaN/Inf propagation policy.

    Args:
        kernel: Callable (input_tensor) -> output_tensor.
        contract: Declared policy.
        base_input_shape: Shape of the base input tensor.
        n_injections: How many positions to inject NaN/Inf at.
        injection_type: "nan", "pos_inf", or "neg_inf".
        propagation_predictor: For IEEE_PROPAGATE, a function mapping a
            bool mask of input exceptional positions to a bool mask of
            expected output exceptional positions. Defaults to elementwise
            identity (suitable for kernels whose output shape matches
            input shape and where exceptional values propagate to the
            same flat position). For reductions, attention, or other
            kernels with non-trivial dataflow, supply a predictor.
        seed: Reproducibility seed.

    Returns:
        NaNPropagationResult.
    """
    if injection_type not in VALID_INJECTION_TYPES:
        raise ValueError(
            f"injection_type must be one of {VALID_INJECTION_TYPES}; "
            f"got {injection_type!r}"
        )
    if seed is not None:
        torch.manual_seed(seed)

    # 1. Build base input
    x = torch.randn(*base_input_shape)
    n_total = x.numel()
    if n_injections > n_total:
        raise ValueError(
            f"n_injections ({n_injections}) cannot exceed total elements "
            f"({n_total})"
        )

    # 2. Inject exceptional values at random flat positions
    flat_indices = torch.randperm(n_total)[:n_injections]
    x_flat = x.flatten().clone()
    if injection_type == "nan":
        x_flat[flat_indices] = float("nan")
    elif injection_type == "pos_inf":
        x_flat[flat_indices] = float("inf")
    elif injection_type == "neg_inf":
        x_flat[flat_indices] = float("-inf")
    x_with_exc = x_flat.reshape(base_input_shape)

    input_exc_mask = torch.isnan(x_flat) | torch.isinf(x_flat)

    # 3. Run kernel once; capture exceptions
    raised_exception: Optional[str] = None
    result: Optional[torch.Tensor] = None
    raised = False
    try:
        result = kernel(x_with_exc)
    except (RuntimeError, ValueError, FloatingPointError) as e:
        raised = True
        raised_exception = f"{type(e).__name__}: {str(e)[:200]}"

    # 4. Build the actual output exceptional-positions mask
    actual_exc_mask: Optional[torch.Tensor] = None
    if result is not None:
        out_flat = result.flatten()
        actual_exc_mask = torch.isnan(out_flat) | torch.isinf(out_flat)

    # 5. Compare against declared policy
    notes = ""
    silent_replacement: Optional[str] = None
    expected_exc_mask: Optional[torch.Tensor] = None

    if contract.declared_policy == "RAISE":
        matches = raised
        if not matches:
            notes = (
                "Declared RAISE; kernel returned values without raising on "
                f"{n_injections} injected exceptional values."
            )

    elif contract.declared_policy == "IEEE_PROPAGATE":
        if raised:
            matches = False
            notes = "Declared IEEE_PROPAGATE; kernel raised."
        else:
            # Predict expected output exceptional positions
            if propagation_predictor is None:
                # Default: elementwise identity
                if result.numel() != n_total:
                    raise ValueError(
                        "Default propagation_predictor only supports kernels "
                        "with matching input/output element count "
                        f"(input={n_total}, output={result.numel()}). "
                        "Provide a custom predictor for shape-changing kernels."
                    )
                expected_exc_mask = input_exc_mask.clone()
            else:
                expected_exc_mask = propagation_predictor(input_exc_mask)

            matches = bool((actual_exc_mask == expected_exc_mask).all())

            if not matches:
                if (
                    not actual_exc_mask.any()
                    and expected_exc_mask.any()
                ):
                    silent_replacement = "input_had_exceptional_output_finite"
                    notes = (
                        f"Declared IEEE_PROPAGATE; input had "
                        f"{int(input_exc_mask.sum().item())} exceptional values, "
                        "output is fully finite. Kernel silently replaced them. "
                        "Compare: Wen et al. on aten::batch_norm (AMD) and "
                        "aten::reshape (Apple Metal)."
                    )
                elif (actual_exc_mask & ~expected_exc_mask).any():
                    silent_replacement = (
                        "exceptional_appeared_at_unexpected_positions"
                    )
                    notes = (
                        "Declared IEEE_PROPAGATE; exceptional values appeared "
                        "at output positions not predicted by the propagation "
                        "rule."
                    )
                else:
                    silent_replacement = (
                        "exceptional_missing_at_expected_positions"
                    )
                    notes = (
                        "Declared IEEE_PROPAGATE; exceptional values are "
                        "missing from output positions where the propagation "
                        "rule predicts them."
                    )

    elif contract.declared_policy == "MASK":
        # MASK is operation-specific; per-operation predicates would extend
        # this verifier. For the reference implementation we record observed
        # behavior without enforcing.
        matches = True
        notes = (
            f"Declared MASK with documented_mask="
            f"{contract.documented_mask!r}; "
            "MASK verification requires an operation-specific predicate. "
            "Observed behavior recorded; not enforced by this reference "
            "verifier."
        )

    else:  # pragma: no cover -- guarded in __post_init__
        raise ValueError(f"Unknown policy: {contract.declared_policy}")

    return NaNPropagationResult(
        contract=contract,
        n_injections=n_injections,
        injection_type=injection_type,
        raised_exception=raised_exception,
        inputs_had_exceptional=bool(input_exc_mask.any()),
        output_has_exceptional=(
            bool(actual_exc_mask.any()) if actual_exc_mask is not None else False
        ),
        expected_exceptional_positions=expected_exc_mask,
        actual_exceptional_positions=actual_exc_mask,
        matches_declared_policy=matches,
        silent_replacement_detected=silent_replacement,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Three-state calibration kernels
# ---------------------------------------------------------------------------


def baseline_kernel(x: torch.Tensor) -> torch.Tensor:
    """Baseline state: returns zeros, eats NaN/Inf entirely.

    Used to confirm the verifier detects total propagation failure.
    """
    return torch.zeros_like(x)


def bad_kernel(x: torch.Tensor) -> torch.Tensor:
    """Bad state: silently replaces NaN/Inf with zero before computing.

    Wen et al.'s aten::batch_norm-on-AMD pattern. Passes basic correctness
    tests on finite inputs; violates a declared IEEE_PROPAGATE contract.
    """
    cleaned = torch.where(
        torch.isnan(x) | torch.isinf(x),
        torch.zeros_like(x),
        x,
    )
    return cleaned + 1.0


def conforming_kernel(x: torch.Tensor) -> torch.Tensor:
    """Conforming state: lets NaN/Inf propagate per IEEE 754.

    Reference implementation for an IEEE_PROPAGATE contract.
    """
    return x + 1.0


# ---------------------------------------------------------------------------
# Pre-built propagation predictors
# ---------------------------------------------------------------------------


def reduction_predictor(input_mask: torch.Tensor) -> torch.Tensor:
    """Predictor for reductions: if any input is exceptional, the output
    (assumed scalar) is exceptional."""
    out = torch.zeros(1, dtype=torch.bool)
    out[0] = bool(input_mask.any())
    return out


def elementwise_predictor(input_mask: torch.Tensor) -> torch.Tensor:
    """Predictor for elementwise kernels: input mask passes through unchanged."""
    return input_mask.clone()
