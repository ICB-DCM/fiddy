from enum import Enum
from typing import Callable, Iterable, Union

import numpy as np


# Currently only 1D arrays are supported.
# TODO Helper methods to support nD arrays, or handle in background.
#      Would need to make changes to:
#      - `.quotient` methods
#      - `.gradient_check` classes
TYPE_DIMENSION = Iterable[int]
TYPE_POINT = np.ndarray
TYPE_OUTPUT = Union[float, int]
TYPE_FUNCTION = Callable[[TYPE_POINT], TYPE_OUTPUT]


class Difference(str, Enum):
    FORWARD = 'forward'
    BACKWARD = 'backward'
    CENTRAL = 'central'


class GradientCheckMethod(str, Enum):
    FORWARD = Difference.FORWARD
    BACKWARD = Difference.BACKWARD
    CENTRAL = Difference.CENTRAL
