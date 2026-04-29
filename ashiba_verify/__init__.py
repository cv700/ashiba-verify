"""ashiba-verify: reference verifiers for ML kernel contract classes.

Implements point-of-use verification protocols for contract classes
defined in "Kernel Contracts: A Specification Language for ML Kernel
Correctness Across Heterogeneous Silicon" (Veit, 2026; arxiv.org/abs/2604.22032).

Currently implemented:
    C-PRC-01  Precision Preservation Under Declared Accumulator
              (via Freivalds-style probabilistic verification)
    C-ORD-01  Reduction-Order Tolerance Bound
    C-CMP-03  Shape-Polymorphism Preservation
    C-EXC-01  NaN/Inf Propagation Semantics
    C-EXC-02  Out-of-Bounds Access Semantics

See https://ashibaresearch.com for context.
"""

from ashiba_verify.freivalds import freivalds_verify, FreivaldsResult
from ashiba_verify.nan_inf import (
    NaNInfContract,
    NaNPropagationResult,
    elementwise_predictor,
    reduction_predictor,
    verify_nan_inf_contract,
)
from ashiba_verify.oob import (
    OOBContract,
    OOBVerificationResult,
    verify_oob_contract,
    run_cross_platform as run_oob_cross_platform,
)
from ashiba_verify.reduction_order import (
    DEFAULT_SCHEDULES,
    PRECISION_EPSILON,
    ReductionOrderContract,
    ReductionOrderResult,
    verify_reduction_order_contract,
)
from ashiba_verify.shape_polymorphism import (
    ShapeClass,
    ShapeContract,
    ShapePolymorphismResult,
    SHAPE_CLASS_GEMM_ANY,
    SHAPE_CLASS_GEMM_MULT16,
    verify_shape_polymorphism,
)

__version__ = "0.5.0"
__all__ = [
    # C-PRC-01
    "freivalds_verify",
    "FreivaldsResult",
    # C-ORD-01
    "ReductionOrderContract",
    "ReductionOrderResult",
    "verify_reduction_order_contract",
    "DEFAULT_SCHEDULES",
    "PRECISION_EPSILON",
    # C-CMP-03
    "ShapeClass",
    "ShapeContract",
    "ShapePolymorphismResult",
    "SHAPE_CLASS_GEMM_ANY",
    "SHAPE_CLASS_GEMM_MULT16",
    "verify_shape_polymorphism",
    # C-EXC-01
    "NaNInfContract",
    "NaNPropagationResult",
    "verify_nan_inf_contract",
    "elementwise_predictor",
    "reduction_predictor",
    # C-EXC-02
    "OOBContract",
    "OOBVerificationResult",
    "verify_oob_contract",
    "run_oob_cross_platform",
]
