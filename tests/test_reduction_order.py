"""Tests for the C-ORD-01 Reduction-Order verifier."""

import pytest
import torch

from ashiba_verify.reduction_order import (
    DEFAULT_SCHEDULES,
    PRECISION_DTYPES,
    PRECISION_EPSILON,
    ReductionOrderContract,
    bad_kernel,
    baseline_kernel,
    conforming_kernel,
    verify_reduction_order_contract,
)


# ---------------------------------------------------------------------------
# Contract construction
# ---------------------------------------------------------------------------


def test_contract_accepts_valid_precisions():
    for p in PRECISION_EPSILON:
        contract = ReductionOrderContract(operation="sum", precision=p)
        assert contract.precision == p


def test_contract_rejects_invalid_precision():
    with pytest.raises(ValueError, match="precision must be one of"):
        ReductionOrderContract(operation="sum", precision="HALF")


def test_precision_epsilon_table():
    """Sanity check: epsilon values match standard machine-epsilon for each
    floating-point format."""
    assert PRECISION_EPSILON["FP64"] == 2.0 ** -52
    assert PRECISION_EPSILON["FP32"] == 2.0 ** -23
    assert PRECISION_EPSILON["BF16"] == 2.0 ** -7
    assert PRECISION_EPSILON["FP16"] == 2.0 ** -10


# ---------------------------------------------------------------------------
# Three-state calibration: FP32 (where the bound is informative)
# ---------------------------------------------------------------------------


def test_conforming_kernel_within_fpna_bound_fp32():
    contract = ReductionOrderContract(operation="sum", precision="FP32")
    result = verify_reduction_order_contract(
        reduction_kernel=conforming_kernel,
        contract=contract,
        n_elements=100_000,
        seed=42,
    )
    assert result.within_bound
    # Some non-zero variation across tree block sizes is expected and bounded
    assert result.max_pairwise_diff < result.fpna_bound


def test_bad_kernel_exceeds_fpna_bound_fp32():
    contract = ReductionOrderContract(operation="sum", precision="FP32")
    result = verify_reduction_order_contract(
        reduction_kernel=bad_kernel,
        contract=contract,
        n_elements=100_000,
        seed=42,
    )
    assert not result.within_bound
    assert result.max_pairwise_diff > result.fpna_bound
    assert "exceed FPNA bound" in result.notes


def test_baseline_kernel_returns_zero_uniformly():
    """Baseline returns 0 regardless of schedule. Pairwise differences are
    zero, so it 'passes' the FPNA bound trivially. This is the documented
    blind spot of reduction-order checking alone -- pair with C-PRC-01
    for full output correctness."""
    contract = ReductionOrderContract(operation="sum", precision="FP32")
    result = verify_reduction_order_contract(
        reduction_kernel=baseline_kernel,
        contract=contract,
        n_elements=100_000,
        seed=42,
    )
    assert result.within_bound
    assert result.max_pairwise_diff == 0.0
    # All schedule outputs are zero
    for v in result.schedule_outputs.values():
        assert v == 0.0


# ---------------------------------------------------------------------------
# Precision regimes (the bound is precision-parameterized)
# ---------------------------------------------------------------------------


def test_fpna_bound_scales_with_precision():
    """At lower precision the FPNA bound is wider; same input, same kernel,
    bound should grow as we move FP32 -> FP16 -> BF16."""
    bound_fp32 = None
    bound_fp16 = None
    bound_bf16 = None

    for precision in ("FP32", "FP16", "BF16"):
        contract = ReductionOrderContract(
            operation="sum", precision=precision
        )
        result = verify_reduction_order_contract(
            reduction_kernel=conforming_kernel,
            contract=contract,
            n_elements=10_000,
            seed=42,
        )
        if precision == "FP32":
            bound_fp32 = result.fpna_bound
        elif precision == "FP16":
            bound_fp16 = result.fpna_bound
        elif precision == "BF16":
            bound_bf16 = result.fpna_bound

    assert bound_fp32 < bound_fp16
    assert bound_fp16 < bound_bf16


def test_bad_kernel_may_pass_at_low_precision():
    """At BF16 the FPNA bound is so wide that the bad kernel's perturbation
    (~0.1 to ~10) may fit inside it. This is documented behavior, not a bug."""
    contract = ReductionOrderContract(operation="sum", precision="BF16")
    result = verify_reduction_order_contract(
        reduction_kernel=bad_kernel,
        contract=contract,
        n_elements=100_000,
        seed=42,
    )
    # bad_kernel injects ~0-10 perturbation; BF16 bound at N=100k, max|x|~4
    # is roughly 100_000 * 7.81e-3 * 4 = 3,124, much larger than 10.
    # So bad_kernel is "within bound" at BF16. This documents the looseness.
    assert result.within_bound  # Note: vacuously, by precision regime


# ---------------------------------------------------------------------------
# FP64 sanity
# ---------------------------------------------------------------------------


def test_fp64_reference_matches_input_sum():
    contract = ReductionOrderContract(operation="sum", precision="FP32")
    result = verify_reduction_order_contract(
        reduction_kernel=conforming_kernel,
        contract=contract,
        n_elements=10_000,
        seed=42,
    )
    # The FP64 reference should be a finite real number close to N(0, sqrt(N))
    assert result.fp64_reference != 0.0
    # FP64 sum of 10k random N(0,1) typically lies in [-300, 300]
    assert abs(result.fp64_reference) < 1000.0


# ---------------------------------------------------------------------------
# Schedule handling
# ---------------------------------------------------------------------------


def test_default_schedules_all_run_for_conforming_kernel():
    contract = ReductionOrderContract(operation="sum", precision="FP32")
    result = verify_reduction_order_contract(
        reduction_kernel=conforming_kernel,
        contract=contract,
        n_elements=10_000,
        seed=42,
    )
    expected_schedule_names = {s["name"] for s in DEFAULT_SCHEDULES}
    assert set(result.schedule_outputs.keys()) == expected_schedule_names


def test_custom_schedules():
    contract = ReductionOrderContract(operation="sum", precision="FP32")
    custom = [
        {"name": "tree_32", "strategy": "tree", "block_size": 32},
        {"name": "tree_512", "strategy": "tree", "block_size": 512},
    ]
    result = verify_reduction_order_contract(
        reduction_kernel=conforming_kernel,
        contract=contract,
        n_elements=10_000,
        schedules=custom,
        seed=42,
    )
    assert set(result.schedule_outputs.keys()) == {"tree_32", "tree_512"}


def test_schedules_with_failures_are_skipped():
    """Schedules the kernel doesn't support should be silently dropped from
    the output set."""

    def selective_kernel(x, schedule):
        if schedule["strategy"] != "torch_default":
            raise NotImplementedError("only supports torch_default")
        return x.sum()

    contract = ReductionOrderContract(operation="sum", precision="FP32")
    result = verify_reduction_order_contract(
        reduction_kernel=selective_kernel,
        contract=contract,
        n_elements=10_000,
        seed=42,
    )
    assert "torch_default" in result.schedule_outputs
    # Other schedules raised NotImplementedError, so they're absent
    assert len(result.schedule_outputs) == 1
    assert "cannot assess pairwise variance" in result.notes


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_seeded_runs_are_deterministic():
    contract = ReductionOrderContract(operation="sum", precision="FP32")
    r1 = verify_reduction_order_contract(
        reduction_kernel=conforming_kernel,
        contract=contract,
        n_elements=10_000,
        seed=123,
    )
    r2 = verify_reduction_order_contract(
        reduction_kernel=conforming_kernel,
        contract=contract,
        n_elements=10_000,
        seed=123,
    )
    assert r1.fp64_reference == r2.fp64_reference
    assert r1.schedule_outputs == r2.schedule_outputs
    assert r1.max_pairwise_diff == r2.max_pairwise_diff


# ---------------------------------------------------------------------------
# FPNA signature
# ---------------------------------------------------------------------------


def test_fpna_signature_for_conforming_kernel():
    """Conforming kernel produces the FPNA signature: non-zero pairwise
    variation that stays within the bound. This is what genuine
    floating-point non-associativity looks like."""
    contract = ReductionOrderContract(operation="sum", precision="FP32")
    result = verify_reduction_order_contract(
        reduction_kernel=conforming_kernel,
        contract=contract,
        n_elements=100_000,
        seed=42,
    )
    if result.max_pairwise_diff > 0:
        assert result.fpna_signature_observed
    # If the conforming kernel happens to produce bit-identical output
    # (possible with specific seeds), the signature is "stronger than
    # required" rather than violated.
