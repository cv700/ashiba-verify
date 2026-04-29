"""Tests for the C-CMP-03 Shape-Polymorphism verifier."""

import pytest
import torch

from ashiba_verify.shape_polymorphism import (
    SHAPE_CLASS_GEMM_ANY,
    SHAPE_CLASS_GEMM_MULT16,
    ShapeClass,
    ShapeContract,
    bad_kernel,
    baseline_kernel,
    conforming_kernel,
    make_matmul_inputs,
    reference_matmul,
    verify_shape_polymorphism,
)


# ---------------------------------------------------------------------------
# Shape class behavior
# ---------------------------------------------------------------------------


def test_gemm_mult16_constraint():
    assert SHAPE_CLASS_GEMM_MULT16.contains((16, 32, 64))
    assert SHAPE_CLASS_GEMM_MULT16.contains((512, 128, 64))
    assert not SHAPE_CLASS_GEMM_MULT16.contains((15, 32, 64))
    assert not SHAPE_CLASS_GEMM_MULT16.contains((16, 32))
    assert not SHAPE_CLASS_GEMM_MULT16.contains((0, 16, 16))


def test_gemm_mult16_sample_returns_n_in_class():
    shapes = SHAPE_CLASS_GEMM_MULT16.sample_shapes(20, seed=42)
    assert len(shapes) == 20
    assert all(SHAPE_CLASS_GEMM_MULT16.contains(s) for s in shapes)


def test_gemm_any_constraint():
    assert SHAPE_CLASS_GEMM_ANY.contains((1, 1, 1))
    assert SHAPE_CLASS_GEMM_ANY.contains((100, 200, 300))
    assert not SHAPE_CLASS_GEMM_ANY.contains((100, 200))


# ---------------------------------------------------------------------------
# Three-state calibration: matmul with GEMM-multiples-of-16
# ---------------------------------------------------------------------------


def test_conforming_kernel_preserves_polymorphism():
    contract = ShapeContract(
        operation="matmul",
        shape_class=SHAPE_CLASS_GEMM_MULT16,
        companion_tolerance={"atol": 1e-4, "rtol": 1e-4},
    )
    benchmarked = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
    result = verify_shape_polymorphism(
        kernel=conforming_kernel,
        reference=reference_matmul,
        contract=contract,
        benchmarked_shapes=benchmarked,
        n_held_out=20,
        seed=42,
    )
    assert result.polymorphism_preserved
    assert result.benchmarked_pass_rate == 1.0
    assert result.held_out_pass_rate == 1.0


def test_bad_kernel_fails_polymorphism():
    contract = ShapeContract(
        operation="matmul",
        shape_class=SHAPE_CLASS_GEMM_MULT16,
        companion_tolerance={"atol": 1e-4, "rtol": 1e-4},
    )
    # bad_kernel hardcodes (1024, 1024, 1024) by default
    benchmarked = [(1024, 1024, 1024)]
    result = verify_shape_polymorphism(
        kernel=bad_kernel,
        reference=reference_matmul,
        contract=contract,
        benchmarked_shapes=benchmarked,
        n_held_out=20,
        seed=42,
    )
    assert not result.polymorphism_preserved
    assert result.benchmarked_pass_rate == 1.0
    assert result.held_out_pass_rate == 0.0
    assert "Sakana" in result.notes or "Lange" in result.notes


def test_baseline_kernel_fails_at_benchmarked():
    contract = ShapeContract(
        operation="matmul",
        shape_class=SHAPE_CLASS_GEMM_MULT16,
        companion_tolerance={"atol": 1e-4, "rtol": 1e-4},
    )
    benchmarked = [(64, 64, 64)]
    result = verify_shape_polymorphism(
        kernel=baseline_kernel,
        reference=reference_matmul,
        contract=contract,
        benchmarked_shapes=benchmarked,
        n_held_out=20,
        seed=42,
    )
    # Baseline returns zeros, fails everywhere
    assert result.benchmarked_pass_rate == 0.0
    assert "cannot meaningfully assess" in result.notes


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_benchmarked_shape_outside_class_raises():
    contract = ShapeContract(
        operation="matmul",
        shape_class=SHAPE_CLASS_GEMM_MULT16,
        companion_tolerance={"atol": 1e-4, "rtol": 1e-4},
    )
    with pytest.raises(ValueError, match="not in declared shape class"):
        verify_shape_polymorphism(
            kernel=conforming_kernel,
            reference=reference_matmul,
            contract=contract,
            benchmarked_shapes=[(15, 16, 16)],  # 15 is not multiple of 16
            n_held_out=10,
            seed=42,
        )


def test_held_out_excludes_benchmarked():
    contract = ShapeContract(
        operation="matmul",
        shape_class=SHAPE_CLASS_GEMM_ANY,
        companion_tolerance={"atol": 1e-4, "rtol": 1e-4},
    )
    benchmarked = [(100, 100, 100)]
    result = verify_shape_polymorphism(
        kernel=conforming_kernel,
        reference=reference_matmul,
        contract=contract,
        benchmarked_shapes=benchmarked,
        n_held_out=20,
        seed=42,
    )
    assert (100, 100, 100) not in result.held_out_shapes


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_seeded_runs_are_deterministic():
    contract = ShapeContract(
        operation="matmul",
        shape_class=SHAPE_CLASS_GEMM_MULT16,
        companion_tolerance={"atol": 1e-4, "rtol": 1e-4},
    )
    benchmarked = [(32, 32, 32)]
    r1 = verify_shape_polymorphism(
        kernel=conforming_kernel,
        reference=reference_matmul,
        contract=contract,
        benchmarked_shapes=benchmarked,
        n_held_out=15,
        seed=42,
    )
    r2 = verify_shape_polymorphism(
        kernel=conforming_kernel,
        reference=reference_matmul,
        contract=contract,
        benchmarked_shapes=benchmarked,
        n_held_out=15,
        seed=42,
    )
    assert r1.held_out_shapes == r2.held_out_shapes
    assert r1.held_out_pass_rate == r2.held_out_pass_rate


# ---------------------------------------------------------------------------
# Custom input builder for non-matmul operations
# ---------------------------------------------------------------------------


def test_custom_input_builder_for_vector_norm():
    """Non-matmul kernel uses a custom shape class + input builder."""

    def _vector_constraint(shape):
        return len(shape) == 1 and shape[0] >= 1

    def _sample_vector_shapes(n, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        return [(int(s),) for s in torch.randint(1, 1024, (n,)).tolist()]

    vector_class = ShapeClass(
        name="vector-any",
        constraint=_vector_constraint,
        sample_shapes=_sample_vector_shapes,
    )

    def _make_vector_inputs(shape, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        return (torch.randn(*shape),)

    def vector_norm(x):
        return torch.norm(x).reshape(1)

    contract = ShapeContract(
        operation="vector_norm",
        shape_class=vector_class,
        companion_tolerance={"atol": 1e-4, "rtol": 1e-4},
    )
    result = verify_shape_polymorphism(
        kernel=vector_norm,
        reference=vector_norm,
        contract=contract,
        benchmarked_shapes=[(100,)],
        n_held_out=15,
        make_inputs=_make_vector_inputs,
        seed=42,
    )
    assert result.polymorphism_preserved


# ---------------------------------------------------------------------------
# Edge case: kernel raises on some shapes
# ---------------------------------------------------------------------------


def test_kernel_that_raises_at_some_shapes_is_caught():
    def fragile_kernel(A, B):
        M, _ = A.shape
        if M > 256:
            raise RuntimeError("Synthetic shape-dependent failure")
        return torch.matmul(A, B)

    contract = ShapeContract(
        operation="matmul",
        shape_class=SHAPE_CLASS_GEMM_MULT16,
        companion_tolerance={"atol": 1e-4, "rtol": 1e-4},
    )
    benchmarked = [(64, 64, 64)]
    result = verify_shape_polymorphism(
        kernel=fragile_kernel,
        reference=reference_matmul,
        contract=contract,
        benchmarked_shapes=benchmarked,
        n_held_out=30,
        seed=42,
    )
    # Some held-out shapes will be > 256 and fail
    assert not result.polymorphism_preserved
    assert len(result.failed_held_out_shapes) > 0
