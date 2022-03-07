import abc
from typing import Any, Callable, Dict, List, Union, Tuple

import numpy as np
import pandas as pd

from .constants import (
    MethodId,
    AnalysisMethod,
    Type,
)

from .step import step

from dataclasses import dataclass, field


@dataclass
class ComputerResult:
    method_id: str
    value: Type.DIRECTIONAL_DERIVATIVE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Computer:
    point: Type.POINT
    # FIXME need to compute directional derivative from user-supplied gradient
    #       method
    direction: Type.DIRECTION
    # FIXME support callable
    size: Type.SIZE
    # Change to callable?
    method: Union[Type.DIRECTIONAL_DERIVATIVE_FUNCTION, MethodId]
    function: Type.FUNCTION
    autorun: bool = True
    completed: bool = False
    results: List[ComputerResult] = field(default_factory=list)
    #value: Type.DIRECTIONAL_DERIVATIVE = None

    def __post_init__(self):
        if isinstance(self.method, MethodId):
            self.method = methods[self.method](
                function=self.function,
            )
            #self.method = get_directional_derivative_method(self.method)(
            #    function=self.function,
            #)
        if self.autorun:
            self()

    def __call__(self):
        value = self.method(
            point=self.point,
            direction=self.direction,
            size=self.size,
        )
        result = ComputerResult(method_id=self.method.id, value=value, metadata={'size': self.size})
        self.results.append(result)
        self.completed = True


# @dataclass
# class ExpectedGradientResult:
#     """
#     Used for gradient checks and tests.
#     Here since `MethodResult` can subclass.
#     These expected values are user-supplied, either
#     or by some gradient-generating method.
#     """
#     # FIXME simply reuse `DirectionalDerivativeResult`?
#     point: Type.POINT
#     # FIXME need to compute directional derivative from user-supplied gradient
#     #       method
#     direction: Type.DIRECTION
#     # FIXME support callable
#     directional_derivative: Type.GRADIENT

# TODO do not inherit from computer
#      define common base class to both?
@dataclass
class ExpectedDirectionalDerivative(Computer):
    size = None
    method = None
    function = None
    autorun = False

    # TODO test
    __call__ = None

    def __post_init__(self):
        if directional_derivative is None:
            raise ValueError('Please provide an expected directional derivative result.')


# @dataclass
# class Analysis:
#     # Change to callable?
#     method: AnalysisMethod
#     result: Any
#
#     def __call__(self, directional_derivative):
#         self.method(directional_derivative=directional_derivative)


@dataclass
class DirectionalDerivative:
    # Each gradient result should have a unique method+step combination.
    id: str
    # FIXME rename to just computers
    pending_computers: List[Computer]
    computers: List[Computer]
    # TODO change to `analysis.Analysis` instead of `Type.ANALYSIS_METHOD`
    analyses: List[Type.ANALYSIS_METHOD]
    # A method that acts on the information in the direction result,
    # including gradient approximation values and post-processing results,
    # to determine whether the gradient was computed successfully.
    # TODO alternatively, can be interpreted as whether the gradient check worked.
    success_checker: Callable[["DirectionalDerivative"], bool] = lambda _: True
    expected_result: ExpectedDirectionalDerivative = None
    # If True, no further computations will occur in this class.
    success: bool = False
    value: Type.SCALAR = None
    completed: bool = False
    # Whether to complete as soon as the success method returns `True`.
    fast: bool = False
    autorun: bool = True

    def __post_init__(self):
        if self.autorun:
            self()

    def __call__(self):
        while not self.completed:
            self.iterate()
        # TODO decide whether to do this here and/or in `run_next_computer` and `check_success`
        self.completed = True

    def iterate(self):
        self.run_next_computer()
        self.run_analyses()
        self.check_success()

    def run_next_computer(self):
        try:
            computer = self.pending_computers.pop(0)
        except IndexError:
            self.completed = True
            return
        computer()
        self.computers.append(computer)

    def get_computer_results(self):
        results = []
        for computer in self.computers:
            results.extend(computer.results)
        return results

    def get_analysis_results(self):
        results = []
        for analysis in self.analyses:
            results.extend(analysis.results)
        return results

    def run_analyses(self):
        for analysis in self.analyses:
            if analysis.only_at_completion and not self.completed:
                continue
            analysis(self)

    def check_success(self):
        if self.success_checker.only_at_completion and not self.completed:
            return
        self.success, value = self.success_checker(self)
        if self.success:
            self.value = value
            if self.fast:
                self.completed = True

class DirectionalDerivativeBase(abc.ABC):
    """Base class for default implementations of directional derivatives.

    Attributes:
        function:
            The function.
    """
    id: MethodId

    def __init__(self, function: Type.FUNCTION):
        """
        Args:
            function:
                The function.
        """
        self.function = function

    @abc.abstractmethod
    def get_points(
        self,
        point: Type.POINT,
        direction: Type.DIRECTION,
        size: Type.SIZE,
    ):
        """Compute the two points used by the method.

        Args:
            See :func:`__call__`.

        Returns:
            (1) the lesser point, (2) the greater point.
        """
        raise NotImplementedError

    def __call__(
        self,
        point: Type.POINT,
        direction: Type.DIRECTION,
        size: Type.SIZE,
    ):
        """Compute the directional derivative.

        Args:
            point:
                The point.
            direction:
                The direction.
            size:
                The step size.

        Returns:
            The directional derivative.
        """
        return self.compute(
            points=self.get_points(
                point=point,
                direction=direction,
                size=size,
            ),
            size=size,
        )

    @abc.abstractmethod
    def compute(self, points: List[Type.POINT]):
        """Compute the directional derivative.

        Args:
            The points to use.

        Returns:
            The directional derivative.
        """
        raise NotImplementedError


class TwoPointSlopeDirectionalDirection(DirectionalDerivativeBase):
    """Derivatives that are similar to a simple `(y1-y0)/h` slope function."""
    def compute(self, points: List[Type.POINT], size: Type.SIZE):
        y0, y1 = self.function(points[0]), self.function(points[1])
        return (y1 - y0)/size


class DefaultForward(TwoPointSlopeDirectionalDirection):
    """The forward difference derivative."""
    id = MethodId.FORWARD

    def get_points(self, point, direction, size):
        x0 = point
        x1 = point + step(direction=direction, size=size)
        return [x0, x1]


class DefaultBackward(TwoPointSlopeDirectionalDirection):
    """The backward difference derivative."""
    id = MethodId.BACKWARD

    def get_points(self, point, direction, size):
        x0 = point - step(direction=direction, size=size)
        x1 = point
        return [x0, x1]


class DefaultCentral(TwoPointSlopeDirectionalDirection):
    """The central difference derivative."""
    id = MethodId.CENTRAL

    def get_points(self, point, direction, size):
        x0 = point - step(direction=direction, size=size/2)
        x1 = point + step(direction=direction, size=size/2)
        return [x0, x1]


methods = {
    method.id: method
    for method in [
        DefaultForward,
        DefaultBackward,
        DefaultCentral,
    ]
}


def standard_basis(point: Type.POINT) -> List[Type.DIRECTION]:
    """Get standard basis (Cartesian/one-hot) vectors.

    Args:
        point:
            The space of this point is used as the space for the standard basis.

    Returns:
        A
    """
    return list(np.eye(point.size, dtype=int))


def get_directions(
    point: Type.POINT = None,
    directions: Union[List[Type.DIRECTION], Dict[str, Type.DIRECTION]] = None,
    ids: List[str] = None,
    indices: List[int] = None,
) -> Tuple[str, Type.DIRECTION]:
    """Get directions from minimal information.

    Args:
        point:
            The standard basis of this point may be used as directions.
        directions:
            The direction vectors.
        ids:
            The direction IDs.
        indices:
            The indices of the standard basis to use as directions.

    Returns:
        (1) Direction IDs, and (2) directions.
    """
    # TODO ensure `direction_ids` type is List[str]?
    # TODO test
    if isinstance(directions, dict):
        if ids is not None:
            raise ValueError('Do not simultaneously specify `directions` as a dictionary, and direction IDs.')
        ids = list(directions.keys())
        directions = list(directions.values())

    if indices is not None and directions is not None:
        raise ValueError('Do not specify indices if directions are specified, as indices will not be used.')

    # Get default directions.
    if point is not None:
        if directions is not None:
            raise NotImplementedError('Please supply only one of `point` and `directions`.')
        directions = standard_basis(point)

    # Apply user indices.
    if indices is not None:
        directions = [
            direction
            for index, direction in enumerate(directions)
            if index in indices
        ]

    # Get default IDs.
    if ids is None:
        ids = [f'direction_{index}' for index in range(len(directions))]

    # Ensure sufficient IDs.
    if len(ids) != len(directions):
        ids = [
            # Old IDs are assumed to be the IDs of the first `len(ids)` indices.
            *ids,
            # Default IDs are generated for the remaining directions.
            *[
                f'direction_{index}'
                for index in range(len(ids), len(directions))
            ]
        ]
    if len(set(ids)) != len(directions):
        raise ValueError('An error occurred related to IDs. Possible clause: duplicate IDs in the supplied `ids`.')

    return ids, directions
