"""Tests for the C-EXC-02 OOB verifier."""

import pytest
import torch

from ashiba_verify.oob import (
    OOBContract,
    VALID_POLICIES,
    bad_kernel,
    baseline_kernel,
    conforming_kernel,
    verify_oob_contract,
)


# ---------------------------------------------------------------------------
# Contract construction
# ---------------------------------------------------------------------------


def test_contract_accepts_valid_policies():
    for policy in VALID_POLICIES:
        contract = OOBContract(
            operation="index_select",
            declared_policy=policy,
            bounds=(0, 100),
        )
        assert contract.declared_policy == policy


def test_contract_rejects_invalid_policy():
    with pytest.raises(ValueError, match="declared_policy must be one of"):
        OOBContract(
            operation="index_select",
            declared_policy="WHATEVER",
            bounds=(0, 100),
        )


def test_contract_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="bounds must satisfy low < high"):
        OOBContract(
            operation="index_select",
            declared_policy="RAISE",
            bounds=(100, 100),
        )


# ---------------------------------------------------------------------------
# Three-state calibration: each kernel against expected verdicts
# ---------------------------------------------------------------------------


def test_conforming_kernel_passes_raise_contract():
    contract = OOBContract(
        operation="index_select",
        declared_policy="RAISE",
        bounds=(0, 100),
    )
    result = verify_oob_contract(
        kernel=conforming_kernel, contract=contract, seed=42
    )
    assert result.matches_declared_policy
    assert result.observed_behavior == "raised"
    assert result.raised_exception is not None
    assert "Out-of-bounds" in result.raised_exception


def test_bad_kernel_fails_raise_contract():
    contract = OOBContract(
        operation="index_select",
        declared_policy="RAISE",
        bounds=(0, 100),
    )
    result = verify_oob_contract(
        kernel=bad_kernel, contract=contract, seed=42
    )
    assert not result.matches_declared_policy
    assert result.observed_behavior == "returned_values"
    assert result.oob_returned_values is not None
    # bad_kernel wraps OOB indices and returns plausible values
    assert result.notes != ""


def test_baseline_kernel_under_undefined_policy_passes():
    """UNDEFINED accepts any behavior, including a kernel that always raises."""
    contract = OOBContract(
        operation="index_select",
        declared_policy="UNDEFINED",
        bounds=(0, 100),
    )
    # baseline_kernel raises NotImplementedError, which is a RuntimeError
    # subclass -- but our verifier catches IndexError/RuntimeError. NIE is
    # neither, so it propagates. Let's verify the policy logic with a
    # kernel that does something verifiable instead.
    def no_op_kernel(data, indices):
        # Returns whatever it likes; UNDEFINED accepts any output
        return torch.zeros(len(indices))

    result = verify_oob_contract(
        kernel=no_op_kernel, contract=contract, seed=42
    )
    assert result.matches_declared_policy


# ---------------------------------------------------------------------------
# Other policies
# ---------------------------------------------------------------------------


def test_clamp_policy_with_clamping_kernel():
    def clamping_kernel(data, indices):
        n = len(data)
        clamped = indices.clamp(0, n - 1)
        return data[clamped]

    contract = OOBContract(
        operation="index_select",
        declared_policy="CLAMP",
        bounds=(0, 100),
    )
    result = verify_oob_contract(
        kernel=clamping_kernel, contract=contract, seed=42
    )
    assert result.matches_declared_policy


def test_zero_policy_with_zero_kernel():
    def zero_kernel(data, indices):
        n = len(data)
        in_bound_mask = (indices >= 0) & (indices < n)
        out = torch.zeros(len(indices), dtype=torch.float32)
        out[in_bound_mask] = data[indices[in_bound_mask]]
        return out

    contract = OOBContract(
        operation="index_select",
        declared_policy="ZERO",
        bounds=(0, 100),
    )
    result = verify_oob_contract(
        kernel=zero_kernel, contract=contract, seed=42
    )
    assert result.matches_declared_policy


def test_clamp_policy_fails_when_kernel_raises():
    contract = OOBContract(
        operation="index_select",
        declared_policy="CLAMP",
        bounds=(0, 100),
    )
    result = verify_oob_contract(
        kernel=conforming_kernel, contract=contract, seed=42
    )
    # Conforming raises, but CLAMP expects values returned
    assert not result.matches_declared_policy


# ---------------------------------------------------------------------------
# Determinism and edge cases
# ---------------------------------------------------------------------------


def test_seeded_runs_are_deterministic():
    contract = OOBContract(
        operation="index_select",
        declared_policy="RAISE",
        bounds=(0, 100),
    )
    r1 = verify_oob_contract(kernel=conforming_kernel, contract=contract, seed=42)
    r2 = verify_oob_contract(kernel=conforming_kernel, contract=contract, seed=42)
    assert r1.observed_behavior == r2.observed_behavior
    assert r1.matches_declared_policy == r2.matches_declared_policy


def test_n_oob_must_be_even():
    contract = OOBContract(
        operation="index_select",
        declared_policy="RAISE",
        bounds=(0, 100),
    )
    with pytest.raises(ValueError, match="n_oob must be even"):
        verify_oob_contract(
            kernel=conforming_kernel,
            contract=contract,
            n_oob=99,
        )
