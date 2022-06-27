"""Methods related to moving along a finite difference direction."""

import numpy as np

from .constants import (
    Type,
)


def step(
    direction: Type.DIRECTION,
    size: Type.SIZE,
):
    """Get a step.

    Args:
        direction:
            The direction to step in.
        size:
            The size of the step.

    Returns:
        The step.
    """
    return direction * size / np.linalg.norm(direction)
