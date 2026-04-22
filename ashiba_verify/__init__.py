"""ashiba-verify: probabilistic GEMM verification for ML kernels across silicon.

Implements Freivalds' algorithm (1979) with PyTorch backends for CUDA, MPS, and ROCm.
See https://ashibaresearch.com for context.
"""

from ashiba_verify.freivalds import freivalds_verify, FreivaldsResult

__version__ = "0.1.0"
__all__ = ["freivalds_verify", "FreivaldsResult"]
