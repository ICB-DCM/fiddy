from typing import Optional

import numpy as np

from .constants import (
    TYPE_FUNCTION,
    TYPE_POINT,
    Difference,
)
from . import difference as difference_methods


def forward(
    function: TYPE_FUNCTION,
    point: TYPE_POINT,
    step: TYPE_POINT,
):
    """Approximate a derivative with a forward difference.

    Args:
        See `compute`.

    Returns:
        The forward difference approximation of the derivative.
    """
    return compute(
        difference=Difference.FORWARD,
        function=function,
        point=point,
        step=step,
    )


def backward(
    function: TYPE_FUNCTION,
    point: TYPE_POINT,
    step: TYPE_POINT,
):
    """Approximate a derivative with a backward difference.

    Args:
        See `compute`.

    Returns:
        The backward difference approximation of the derivative.
    """
    return compute(
        difference=Difference.BACKWARD,
        function=function,
        point=point,
        step=step,
    )


def central(
    function: TYPE_FUNCTION,
    point: TYPE_POINT,
    step: TYPE_POINT,
):
    """Approximate a derivative with a central difference.

    Args:
        See `compute`.

    Returns:
        The central difference approximation of the derivative.
    """
    return compute(
        difference=Difference.CENTRAL,
        function=function,
        point=point,
        step=step,
    )


def compute(
    difference: Difference,
    function: TYPE_FUNCTION,
    point: TYPE_POINT,
    step: TYPE_POINT,
):
    """Approximate a derivative with a finite difference.

    Args:
        See `difference.forward`.
        difference:
            The type of finite difference to use.

    Returns:
        The finite difference approximation of the derivative.
    """
    size = np.linalg.norm(step)
    method = getattr(difference_methods, difference.value)
    numerator = method(function=function, point=point, step=step)
    quotient = numerator / size
    return quotient
