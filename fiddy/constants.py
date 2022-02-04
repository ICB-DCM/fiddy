from enum import Enum
from typing import Callable, Union

import numpy as np

__all__ = [
    "TYPE_DIMENSION",
    "TYPE_FUNCTION",
    "TYPE_OUTPUT",
    "TYPE_POINT",
    "Difference",
    "GradientCheckMethod",
]

# Currently only 1D arrays are supported.
# TODO Helper methods to support nD arrays, or handle in background.
#      Would need to make changes to:
#      - `.quotient` methods
#      - `.gradient_check` classes
#      ... or just flatten
TYPE_DIMENSION = int
TYPE_OUTPUT = Union[float, int]
TYPE_POINT = np.ndarray
TYPE_FUNCTION = Callable[[TYPE_POINT], TYPE_OUTPUT]


class Difference(str, Enum):
    BACKWARD = "backward"
    CENTRAL = "central"
    FORWARD = "forward"


class GradientCheckMethod(str, Enum):
    BACKWARD = Difference.BACKWARD
    CENTRAL = Difference.CENTRAL
    FORWARD = Difference.FORWARD
