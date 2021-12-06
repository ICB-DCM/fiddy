from enum import Enum
from typing import Callable, Iterable, Union

import numpy as np


TYPE_DIMENSION = Iterable[int]
TYPE_POINT = np.ndarray
# TODO generalize to np.ndarray output?
#      would need to make changes to `.quotient` methods
TYPE_FUNCTION = Callable[[TYPE_POINT], Union[float, int]]


class Difference(str, Enum):
    FORWARD = 'forward'
    BACKWARD = 'backward'
    CENTRAL = 'central'
