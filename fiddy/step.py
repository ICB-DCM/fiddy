# from typing import Iterable, Tuple, Union

import numpy as np

from .constants import (
    TYPE_DIMENSION,
    TYPE_POINT,
)


def dstep(
    point: TYPE_POINT,
    dimension: TYPE_DIMENSION,
    size: float,
) -> TYPE_POINT:
    """Generate a step from a step size and a single dimension to step along.

    Args:
        point:
            The point to step from.
        dimension:
            The dimension to step along.
        size:
            The size of the step.

    Returns:
        The step, that can be added to the point.
    """
    # return one_hot_array(
    #     shape=point.shape,
    #     index=dimension,
    #     value=size,
    # )
    array = np.zeros(point.shape)
    array[dimension] = size
    return array


# def ndstep(
#     point: TYPE_POINT,
#     dimensions: Iterable[TYPE_DIMENSION],
#     sizes: Iterable[float],
# ) -> TYPE_POINT:
#     """Generate a step.
#
#     Args:
#         point:
#             The point to step from.
#         dimensions:
#             The dimensions to step along.
#         sizes:
#             The size of the steps, one for each dimension.
#
#     Returns:
#         The step, that can be added to the point.
#     """
#     array = np.zeros(point.shape)
#     for dimension, size in zip(dimensions, sizes):
#         array[dimension] = size
#     return array
#
#
# def one_hot_array(
#     shape: TYPE_POINT,
#     index: TYPE_DIMENSION,
#     value: float,
# ) -> TYPE_POINT:
#     """Create a one-hot array.
#
#     Args:
#         shape:
#             The shape of the array.
#         index:
#             The index of the hot element.
#         value:
#             The value of the hot element.
#
#     Returns:
#         The one-hot array.
#     """
#     array = np.zeros(shape)
#     array[index] = value
#     return array
