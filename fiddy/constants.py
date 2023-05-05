from enum import Enum
from typing import Any, Callable, Union

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "Type",
    "MethodId",
]


class Type:
    """Type annotation variables."""

    # The size is applied as the Euclidean distance in the
    # specified direction, between the point and the stepped point
    SCALAR = np.float64
    ARRAY = NDArray[SCALAR]
    DIRECTION = ARRAY
    SIZE = SCALAR  # strictly positive TODO enforce?
    POINT = ARRAY
    DIRECTIONAL_DERIVATIVE = ARRAY
    DERIVATIVE = NDArray[DIRECTIONAL_DERIVATIVE]
    # Currently only supports scalar-valued functions with
    # vector input of arbitrary dimension
    FUNCTION_OUTPUT = ARRAY
    FUNCTION = Callable[[POINT], FUNCTION_OUTPUT]
    DERIVATIVE_FUNCTION = Callable[[POINT], DERIVATIVE]
    DIRECTIONAL_DERIVATIVE_FUNCTION = Callable[[POINT], DIRECTIONAL_DERIVATIVE]
    # TODO rename analysis and success to e.g.
    #      - "ANALYSE_DIRECTIONAL_DERIVATIVE_METHOD" and
    #      - "ANALYSE_DERIVATIVE_METHOD" and
    ANALYSIS_METHOD = Callable[
        ["directional_derivative.DirectionalDerivative"], Any
    ]
    SUCCESS_CHECKER = Callable[["derivative.Derivative"], Union[bool, Any]]


# FIXME rename, since this can be the name of the base class in `derivative.py`
# FIXME use Difference instead, then i.e. Extrapolation too
class MethodId(str, Enum):
    """Finite different method IDs."""

    BACKWARD = "backward"
    CENTRAL = "central"
    FORWARD = "forward"
    # richardson
    # five point?
    # TODO separate enum for "order" of method?
    #      e.g. for higher-order derivatives?
    HYBRID = "hybrid"


# class AnalysisMethod(str, Enum):
#    ABSOLUTE_ERROR = "absolute_error"
#    RELATIVE_ERROR = "relative_error"
# =======


## TODO remove? redundant
# class GradientCheckMethod(str, Enum):
#    BACKWARD = MethodId.BACKWARD
#    CENTRAL = MethodId.CENTRAL
#    FORWARD = MethodId.FORWARD
#    HYBRID = MethodId.HYBRID
##>>>>>>> origin/main


EPSILON = 1e-5
