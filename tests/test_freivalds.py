"""Tests for the Freivalds verifier.

Runs on whatever device is available (prefers CUDA, then MPS, then CPU).
"""

import pytest
import torch

from ashiba_verify import freivalds_verify, FreivaldsResult


def get_device():
    """Prefer CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def device():
    return get_device()


class TestCorrect:
    """Verifier passes on correct matmuls."""

    def test_small_fp32(self, device):
        A = torch.randn(32, 16, device=device, dtype=torch.float32)
        B = torch.randn(16, 24, device=device, dtype=torch.float32)
        C = A @ B
        assert freivalds_verify(A, B, C, k=10) is True

    def test_medium_fp32(self, device):
        A = torch.randn(512, 256, device=device, dtype=torch.float32)
        B = torch.randn(256, 128, device=device, dtype=torch.float32)
        C = A @ B
        assert freivalds_verify(A, B, C, k=10) is True

    def test_bf16(self, device):
        if device.type == "mps":
            pytest.skip("BF16 matmul not fully supported on MPS yet")
        A = torch.randn(256, 128, device=device, dtype=torch.bfloat16)
        B = torch.randn(128, 64, device=device, dtype=torch.bfloat16)
        C = A @ B
        assert freivalds_verify(A, B, C, k=10) is True

    def test_large_fp16(self, device):
        if device.type == "cpu":
            pytest.skip("FP16 on CPU is slow")
        A = torch.randn(1024, 512, device=device, dtype=torch.float16)
        B = torch.randn(512, 256, device=device, dtype=torch.float16)
        C = A @ B
        assert freivalds_verify(A, B, C, k=10) is True


class TestCorruption:
    """Verifier catches corrupted matmuls."""

    @pytest.mark.xfail(
        reason="Norm-based tolerance admits single-element corruptions below the "
               "sensitivity floor (rtol * ||cr||_inf). See TRACES.md Entry 1 Bug 2A "
               "for the tradeoff discussion. Hybrid tolerance in v0.2 should resolve."
    )
    def test_single_element_corruption(self, device):
        A = torch.randn(256, 128, device=device, dtype=torch.float32)
        B = torch.randn(128, 64, device=device, dtype=torch.float32)
        C_correct = A @ B

        C_corrupt = C_correct.clone()
        C_corrupt[0, 0] += 0.01  # below the norm-based sensitivity floor; xfail

        assert freivalds_verify(A, B, C_corrupt, k=20) is False

    def test_random_noise_corruption(self, device):
        A = torch.randn(256, 128, device=device, dtype=torch.float32)
        B = torch.randn(128, 64, device=device, dtype=torch.float32)
        C_correct = A @ B

        C_corrupt = C_correct + 0.001 * torch.randn_like(C_correct)

        assert freivalds_verify(A, B, C_corrupt, k=20) is False

    def test_scaled_output_corruption(self, device):
        """Multiplying the output by a constant ≠ 1 should be caught."""
        A = torch.randn(128, 64, device=device, dtype=torch.float32)
        B = torch.randn(64, 32, device=device, dtype=torch.float32)
        C_correct = A @ B

        C_corrupt = C_correct * 1.001  # 0.1% scaling

        assert freivalds_verify(A, B, C_corrupt, k=20) is False


class TestShapeValidation:
    """Shape mismatches should raise."""

    def test_wrong_inner_dim(self, device):
        A = torch.randn(16, 8, device=device)
        B = torch.randn(4, 8, device=device)  # wrong inner dim
        C = torch.zeros(16, 8, device=device)
        with pytest.raises(ValueError, match="Inner dimensions"):
            freivalds_verify(A, B, C, k=5)

    def test_wrong_output_shape(self, device):
        A = torch.randn(16, 8, device=device)
        B = torch.randn(8, 4, device=device)
        C = torch.zeros(16, 5, device=device)  # wrong output width
        with pytest.raises(ValueError, match="Output shape"):
            freivalds_verify(A, B, C, k=5)

    def test_non_2d_input_raises(self, device):
        A = torch.randn(16, 8, 2, device=device)  # 3D
        B = torch.randn(8, 4, device=device)
        C = torch.zeros(16, 4, device=device)
        with pytest.raises(ValueError, match="2D matrices"):
            freivalds_verify(A, B, C, k=5)


class TestReturnResult:
    """return_result=True returns a FreivaldsResult dataclass."""

    def test_pass_result(self, device):
        A = torch.randn(64, 32, device=device)
        B = torch.randn(32, 16, device=device)
        C = A @ B
        result = freivalds_verify(A, B, C, k=5, return_result=True)
        assert isinstance(result, FreivaldsResult)
        assert result.passed is True
        assert result.k == 5
        assert result.false_positive_probability == pytest.approx(1.0 / 32.0)

    def test_fail_result(self, device):
        A = torch.randn(64, 32, device=device)
        B = torch.randn(32, 16, device=device)
        C = A @ B
        C_bad = C + 0.01
        result = freivalds_verify(A, B, C_bad, k=5, return_result=True)
        assert result.passed is False
        assert result.max_residual > 0


class TestReproducibility:
    """Seed produces reproducible results."""

    def test_seed_reproducibility(self, device):
        A = torch.randn(64, 32, device=device)
        B = torch.randn(32, 16, device=device)
        C = A @ B
        r1 = freivalds_verify(A, B, C, k=5, seed=42, return_result=True)
        r2 = freivalds_verify(A, B, C, k=5, seed=42, return_result=True)
        assert r1.max_residual == r2.max_residual
