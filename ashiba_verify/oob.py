"""C-EXC-02: Out-of-Bounds Access Semantics.

Verify that an indexing kernel honors its declared out-of-bound (OOB)
access policy.

A kernel claims one of four policies:
    RAISE     - raises on out-of-bound access
    CLAMP     - clamps the index to the valid range; documented behavior
    ZERO      - returns zero for out-of-bound positions; documented behavior
    UNDEFINED - explicit undefined behavior; any output is acceptable

The verifier constructs an index array with both in-bound and intentionally
out-of-bound indices, runs the kernel once, and compares observed behavior
to the declared policy. The reference is PyTorch / NumPy CPU semantics.

Empirical context. Wen et al. (2025), "Mind the Gap," report approximately
1,700 fewer out-of-bound exceptions on AMD MI300X than on the NVIDIA H200
baseline and Intel MAX 1100 baseline for the same inputs. That is roughly
1,700 cases where NVIDIA and Intel raise and AMD silently returns a
non-erroring value. The returned values were not documented as clamp,
zero, or any declared policy. C-EXC-02 makes that gap nameable.

Usage. This module is a point-of-use test protocol. Invoke once against
a candidate kernel to obtain a single-shot pass/fail verdict. It is not
a continuous probe.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch


VALID_POLICIES = ("RAISE", "CLAMP", "ZERO", "UNDEFINED")


# ---------------------------------------------------------------------------
# Contract and result types
# ---------------------------------------------------------------------------


@dataclass
class OOBContract:
    """Declares the out-of-bound policy a kernel claims to honor.

    Attributes:
        operation: Name of the indexing op being verified
            (e.g., "gather", "scatter", "index_select", "embedding").
        declared_policy: One of "RAISE", "CLAMP", "ZERO", "UNDEFINED".
        bounds: Inclusive low / exclusive high range of valid indices.
    """

    operation: str
    declared_policy: str
    bounds: Tuple[int, int]

    def __post_init__(self):
        if self.declared_policy not in VALID_POLICIES:
            raise ValueError(
                f"declared_policy must be one of {VALID_POLICIES}; "
                f"got {self.declared_policy!r}"
            )
        low, high = self.bounds
        if low >= high:
            raise ValueError(
                f"bounds must satisfy low < high; got ({low}, {high})"
            )


@dataclass
class OOBVerificationResult:
    """Outcome of a single OOB verification run.

    Attributes:
        contract: The contract that was verified.
        n_in_bound: Number of in-bound indices in the test.
        n_oob: Number of out-of-bound indices in the test.
        observed_behavior: Either "raised" or "returned_values".
        raised_exception: String of the exception (truncated) if raised.
        oob_returned_values: Values the kernel returned at OOB positions
            (None if it raised or did not return a tensor).
        matches_declared_policy: True iff observed behavior matches the
            declared policy.
        notes: Human-readable explanation of mismatches.
    """

    contract: OOBContract
    n_in_bound: int
    n_oob: int
    observed_behavior: str
    raised_exception: Optional[str]
    oob_returned_values: Optional[torch.Tensor]
    matches_declared_policy: bool
    notes: str


# ---------------------------------------------------------------------------
# Verification logic
# ---------------------------------------------------------------------------


def verify_oob_contract(
    kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    contract: OOBContract,
    n_in_bound: int = 1000,
    n_oob: int = 100,
    seed: Optional[int] = None,
) -> OOBVerificationResult:
    """Test whether a kernel honors its declared OOB-access policy.

    Constructs a mixed index array (in-bound + intentional OOB), runs the
    kernel once, and compares observed behavior to the declared policy.

    Args:
        kernel: Callable (data, indices) -> output. Performs the indexing
            operation declared in the contract.
        contract: The OOB contract to verify.
        n_in_bound: Number of in-bound indices to include.
        n_oob: Number of out-of-bound indices (split half above, half below).
            Must be even.
        seed: Reproducibility seed for index sampling.

    Returns:
        OOBVerificationResult.
    """
    if n_oob % 2 != 0:
        raise ValueError(f"n_oob must be even; got {n_oob}")

    if seed is not None:
        torch.manual_seed(seed)

    low, high = contract.bounds

    # 1. Construct mixed index array: in-bound + OOB above + OOB below
    in_bound = torch.randint(low, high, (n_in_bound,))
    oob_above = torch.randint(high, high + 100, (n_oob // 2,))
    oob_below = torch.randint(low - 100, low, (n_oob // 2,))
    indices = torch.cat([in_bound, oob_above, oob_below])

    # 2. Build a data tensor to index into
    data = torch.arange(low, high, dtype=torch.float32)

    # 3. Run the kernel once; catch indexing exceptions
    raised_exception: Optional[str] = None
    result: Optional[torch.Tensor] = None
    try:
        result = kernel(data, indices)
        observed = "returned_values"
    except (IndexError, RuntimeError) as e:
        observed = "raised"
        raised_exception = f"{type(e).__name__}: {str(e)[:200]}"

    # 4. Compare observed behavior to declared policy
    notes = ""
    if contract.declared_policy == "RAISE":
        matches = observed == "raised"
        if not matches:
            notes = (
                "Declared RAISE; kernel returned values without raising on "
                f"{n_oob} out-of-bound indices."
            )

    elif contract.declared_policy == "CLAMP":
        if observed != "returned_values":
            matches = False
            notes = "Declared CLAMP; kernel raised."
        else:
            clamped = torch.clamp(indices, low, high - 1)
            expected = data[clamped - low]
            matches = bool(torch.equal(result, expected))
            if not matches:
                notes = (
                    "Declared CLAMP; output does not match the clamped-index "
                    "reference (PyTorch CPU)."
                )

    elif contract.declared_policy == "ZERO":
        if observed != "returned_values":
            matches = False
            notes = "Declared ZERO; kernel raised."
        else:
            in_bound_mask = (indices >= low) & (indices < high)
            expected = torch.zeros(len(indices), dtype=torch.float32)
            expected[in_bound_mask] = data[indices[in_bound_mask] - low]
            matches = bool(torch.equal(result, expected))
            if not matches:
                notes = (
                    "Declared ZERO; output does not match (in-bound = data, "
                    "OOB = 0) reference."
                )

    elif contract.declared_policy == "UNDEFINED":
        # By contract, any output is acceptable.
        matches = True
        notes = "Declared UNDEFINED; any output accepted by the contract."

    else:  # pragma: no cover -- guarded in __post_init__
        raise ValueError(f"Unknown policy: {contract.declared_policy}")

    # 5. Capture the OOB-position output for diagnostics, when available.
    oob_values: Optional[torch.Tensor] = None
    if result is not None and len(result) >= n_in_bound + n_oob:
        oob_values = result[n_in_bound : n_in_bound + n_oob].clone()

    return OOBVerificationResult(
        contract=contract,
        n_in_bound=n_in_bound,
        n_oob=n_oob,
        observed_behavior=observed,
        raised_exception=raised_exception,
        oob_returned_values=oob_values,
        matches_declared_policy=matches,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Three-state calibration kernels
# (Provided to exercise the verifier itself; see paper Section 5.3)
# ---------------------------------------------------------------------------


def baseline_kernel(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Baseline state: always raises, even for in-bound indices.

    Used to confirm the verifier detects complete functional failure.
    Fails any contract whose declared policy is not RAISE or UNDEFINED.
    """
    raise NotImplementedError("Baseline kernel: not implemented.")


def bad_kernel(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Bad state: silently wraps OOB indices via modulo.

    Returns plausible values for OOB positions. Passes simple in-bound-only
    smoke tests; violates a declared RAISE contract by silently accepting
    OOB indices.
    """
    safe = indices.clamp(min=0) % len(data)
    return data[safe]


def conforming_kernel(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Conforming state: declares RAISE policy and raises on any OOB.

    Reference implementation for a contract whose declared policy is RAISE.
    """
    n = len(data)
    if (indices < 0).any() or (indices >= n).any():
        raise IndexError("Out-of-bounds indices detected")
    return data[indices]


# ---------------------------------------------------------------------------
# Cross-platform reproduction helper
# ---------------------------------------------------------------------------


def run_cross_platform(
    kernel_factory: Callable[[torch.device], Callable],
    contract: OOBContract,
    devices: Tuple[str, ...] = ("cpu", "cuda", "mps"),
    n_in_bound: int = 1000,
    n_oob: int = 100,
    seed: Optional[int] = 42,
) -> Dict[str, OOBVerificationResult]:
    """Run the same OOB test on multiple devices.

    Reproduces cross-platform divergence in OOB handling. With a single
    `bad_kernel` declared as RAISE, the result table typically shows CUDA
    and CPU raising while MPS may silently return values, illustrating
    the empirical gap reported by Wen et al.

    Args:
        kernel_factory: Callable (device) -> kernel. Allows the caller to
            build a device-specific kernel that uses tensors on `device`.
        contract: OOB contract.
        devices: Devices to attempt. Unavailable devices are skipped.
        n_in_bound, n_oob, seed: Forwarded to verify_oob_contract.

    Returns:
        Mapping of device name to OOBVerificationResult.
    """
    results: Dict[str, OOBVerificationResult] = {}
    for device_name in devices:
        try:
            device = torch.device(device_name)
            torch.tensor([1.0], device=device)  # availability probe
        except (RuntimeError, ValueError, AssertionError):
            continue

        kernel = kernel_factory(device)
        result = verify_oob_contract(
            kernel=kernel,
            contract=contract,
            n_in_bound=n_in_bound,
            n_oob=n_oob,
            seed=seed,
        )
        results[device_name] = result

    return results
