"""Tests for the C-EXC-01 NaN/Inf verifier."""

import pytest
import torch

from ashiba_verify.nan_inf import (
    NaNInfContract,
    VALID_INJECTION_TYPES,
    VALID_POLICIES,
    bad_kernel,
    baseline_kernel,
    conforming_kernel,
    reduction_predictor,
    verify_nan_inf_contract,
)


# ---------------------------------------------------------------------------
# Contract construction
# ---------------------------------------------------------------------------


def test_contract_accepts_valid_policies():
    for policy in VALID_POLICIES:
        kwargs = {"operation": "elementwise_add", "declared_policy": policy}
        if policy == "MASK":
            kwargs["documented_mask"] = "zero"
        contract = NaNInfContract(**kwargs)
        assert contract.declared_policy == policy


def test_contract_rejects_invalid_policy():
    with pytest.raises(ValueError, match="declared_policy must be one of"):
        NaNInfContract(operation="x", declared_policy="WHATEVER")


def test_mask_policy_requires_documented_mask():
    with pytest.raises(ValueError, match="MASK policy requires"):
        NaNInfContract(operation="x", declared_policy="MASK")


def test_mask_policy_accepts_documented_mask():
    contract = NaNInfContract(
        operation="x", declared_policy="MASK", documented_mask="zero"
    )
    assert contract.documented_mask == "zero"


# ---------------------------------------------------------------------------
# Three-state calibration
# ---------------------------------------------------------------------------


def test_conforming_kernel_passes_ieee_propagate():
    contract = NaNInfContract(
        operation="elementwise_add",
        declared_policy="IEEE_PROPAGATE",
    )
    result = verify_nan_inf_contract(
        kernel=conforming_kernel,
        contract=contract,
        base_input_shape=(100,),
        seed=42,
    )
    assert result.matches_declared_policy
    assert result.silent_replacement_detected is None


def test_bad_kernel_fails_ieee_propagate():
    contract = NaNInfContract(
        operation="elementwise_add",
        declared_policy="IEEE_PROPAGATE",
    )
    result = verify_nan_inf_contract(
        kernel=bad_kernel,
        contract=contract,
        base_input_shape=(100,),
        seed=42,
    )
    assert not result.matches_declared_policy
    assert (
        result.silent_replacement_detected
        == "input_had_exceptional_output_finite"
    )
    assert "Wen et al." in result.notes


def test_baseline_kernel_fails_ieee_propagate():
    contract = NaNInfContract(
        operation="elementwise_add",
        declared_policy="IEEE_PROPAGATE",
    )
    result = verify_nan_inf_contract(
        kernel=baseline_kernel,
        contract=contract,
        base_input_shape=(100,),
        seed=42,
    )
    assert not result.matches_declared_policy
    assert (
        result.silent_replacement_detected
        == "input_had_exceptional_output_finite"
    )


# ---------------------------------------------------------------------------
# Different injection types
# ---------------------------------------------------------------------------


def test_pos_inf_injection_propagates():
    contract = NaNInfContract(
        operation="elementwise_add",
        declared_policy="IEEE_PROPAGATE",
    )
    result = verify_nan_inf_contract(
        kernel=conforming_kernel,
        contract=contract,
        injection_type="pos_inf",
        base_input_shape=(100,),
        seed=42,
    )
    assert result.matches_declared_policy


def test_neg_inf_injection_propagates():
    contract = NaNInfContract(
        operation="elementwise_add",
        declared_policy="IEEE_PROPAGATE",
    )
    result = verify_nan_inf_contract(
        kernel=conforming_kernel,
        contract=contract,
        injection_type="neg_inf",
        base_input_shape=(100,),
        seed=42,
    )
    assert result.matches_declared_policy


# ---------------------------------------------------------------------------
# RAISE policy
# ---------------------------------------------------------------------------


def test_raise_policy_with_raising_kernel():
    def raising_kernel(x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise FloatingPointError("Exceptional value in input")
        return x + 1.0

    contract = NaNInfContract(
        operation="elementwise_add",
        declared_policy="RAISE",
    )
    result = verify_nan_inf_contract(
        kernel=raising_kernel, contract=contract, seed=42
    )
    assert result.matches_declared_policy
    assert result.raised_exception is not None
    assert "Exceptional value" in result.raised_exception


def test_raise_policy_fails_when_kernel_does_not_raise():
    contract = NaNInfContract(
        operation="elementwise_add",
        declared_policy="RAISE",
    )
    result = verify_nan_inf_contract(
        kernel=conforming_kernel, contract=contract, seed=42
    )
    assert not result.matches_declared_policy


# ---------------------------------------------------------------------------
# Custom propagation predictor
# ---------------------------------------------------------------------------


def test_custom_predictor_for_reduction():
    """Reduction over a tensor: any NaN input -> entire output is NaN."""

    def reduction_kernel(x):
        return x.sum().reshape(1)

    contract = NaNInfContract(
        operation="sum_reduction",
        declared_policy="IEEE_PROPAGATE",
    )
    result = verify_nan_inf_contract(
        kernel=reduction_kernel,
        contract=contract,
        propagation_predictor=reduction_predictor,
        base_input_shape=(100,),
        seed=42,
    )
    assert result.matches_declared_policy


def test_default_predictor_rejects_shape_changing_kernel():
    """Without a custom predictor, kernels that change element count must
    not silently use the elementwise default."""

    def reducing_kernel(x):
        return x.sum().reshape(1)

    contract = NaNInfContract(
        operation="sum_reduction",
        declared_policy="IEEE_PROPAGATE",
    )
    with pytest.raises(
        ValueError, match="Default propagation_predictor only supports"
    ):
        verify_nan_inf_contract(
            kernel=reducing_kernel,
            contract=contract,
            base_input_shape=(100,),
            seed=42,
        )


# ---------------------------------------------------------------------------
# MASK policy (recorded only)
# ---------------------------------------------------------------------------


def test_mask_policy_records_but_does_not_enforce():
    contract = NaNInfContract(
        operation="batch_norm_with_mask",
        declared_policy="MASK",
        documented_mask="zero",
    )
    result = verify_nan_inf_contract(
        kernel=bad_kernel, contract=contract, seed=42
    )
    # MASK reference implementation is non-enforcing
    assert result.matches_declared_policy
    assert "MASK verification requires" in result.notes


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_invalid_injection_type():
    contract = NaNInfContract(
        operation="x", declared_policy="IEEE_PROPAGATE"
    )
    with pytest.raises(ValueError, match="injection_type must be one of"):
        verify_nan_inf_contract(
            kernel=conforming_kernel,
            contract=contract,
            injection_type="quiet_nan",
        )


def test_n_injections_too_large():
    contract = NaNInfContract(
        operation="x", declared_policy="IEEE_PROPAGATE"
    )
    with pytest.raises(ValueError, match="cannot exceed total elements"):
        verify_nan_inf_contract(
            kernel=conforming_kernel,
            contract=contract,
            base_input_shape=(10,),
            n_injections=100,
        )


def test_seeded_runs_are_deterministic():
    contract = NaNInfContract(
        operation="elementwise_add",
        declared_policy="IEEE_PROPAGATE",
    )
    r1 = verify_nan_inf_contract(
        kernel=conforming_kernel, contract=contract, seed=123
    )
    r2 = verify_nan_inf_contract(
        kernel=conforming_kernel, contract=contract, seed=123
    )
    assert r1.matches_declared_policy == r2.matches_declared_policy
    assert r1.inputs_had_exceptional == r2.inputs_had_exceptional


def test_valid_injection_types_constant():
    """Sanity check: every documented injection type is actually accepted."""
    contract = NaNInfContract(
        operation="x", declared_policy="IEEE_PROPAGATE"
    )
    for itype in VALID_INJECTION_TYPES:
        result = verify_nan_inf_contract(
            kernel=conforming_kernel,
            contract=contract,
            base_input_shape=(50,),
            injection_type=itype,
            seed=42,
        )
        assert result.matches_declared_policy
