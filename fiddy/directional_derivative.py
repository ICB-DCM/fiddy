import abc
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np

from . import directional_derivative
from .constants import EPSILON, MethodId, Type
from .step import step


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
    # value: Type.DIRECTIONAL_DERIVATIVE = None
    relative_size: bool = False

    def __post_init__(self):
        if isinstance(self.method, MethodId):
            self.method = methods[self.method](
                function=self.function,
            )
            # self.method = get_directional_derivative_method(self.method)(
            #    function=self.function,
            # )
        if self.autorun:
            self()

    def get_size(self):
        if not self.relative_size:
            return self.size

        # If relative, project point onto direction as scaling factor for size
        unit_direction = self.direction / np.linalg.norm(self.direction)
        # TODO add some epsilon to size?
        size = np.dot(self.point, unit_direction) * self.size
        if size == 0:
            warnings.warn(
                "Point has no component in this direction. "
                "Set `Computer.relative_size=False` to avoid this. "
                f"Using default small step size `fiddy.EPSILON`: {EPSILON}",
                stacklevel=1,
            )
            size = EPSILON
        return size

    def __call__(self):
        value = self.method(
            point=self.point,
            direction=self.direction,
            size=self.get_size(),
        )
        result = ComputerResult(
            method_id=self.method.id,
            value=value,
            metadata={"size": self.get_size(), "size_absolute": self.size},
        )
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
            raise ValueError(
                "Please provide an expected directional derivative result."
            )


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
    direction: Type.DIRECTION
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
        """Construct.

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
            The points at which the function will be evaluated.
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
            direction=direction,
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

    def compute(self, points: List[Type.POINT], size: Type.SIZE, **kwargs):
        y0, y1 = self.function(points[0]), self.function(points[1])
        return (y1 - y0) / size


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
        x0 = point - step(direction=direction, size=size / 2)
        x1 = point + step(direction=direction, size=size / 2)
        return [x0, x1]


class DefaultRichardson(DirectionalDerivativeBase):
    r"""The Richardson extrapolation method.

    Based on https://doi.org/10.48550/arXiv.2110.04335

    Given some initial step size `h` and some order `n`, terms are
    computed as

    .. math::

        A_{i,j} = \mathrm{Central\,Difference}\left(\mathrmP{step size}= \frac{h}{2^{i-1}} \right)

    if `j = 1`, and

    .. math::

        A_{i,j} = \frac{4^{j-1} A_{i,j-1} - A_{i-1,j-1}}{4^{j-1} - 1}

    otherwise.

    The derivative is given by `A` at `i=n`, `j=n`.

    Some basic caching is used, which is reset when a new derivative is requested.
    """

    id = MethodId.RICHARDSON
    order = 4
    # TODO change order to some tolerance?

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.central = DefaultCentral(function=self.function)

        self.reset_cache()

    def reset_cache(self):
        self.cache = {}

    def get_points(self, point, direction, size):
        return point

    def get_term(self, i, j, **kwargs):
        if (i, j) in self.cache:
            return self.cache[(i, j)]

        if j == 1:
            size = kwargs["size"] / (2 ** (i - 1))
            term = self.central(
                size=size, **{k: v for k, v in kwargs.items() if k != "size"}
            )
            self.cache[(i, j)] = term
            return term

        term = (
            4 ** (j - 1) * self.get_term(i=i, j=j - 1, **kwargs)
            - self.get_term(i=i - 1, j=j - 1, **kwargs)
        ) / (4 ** (j - 1) - 1)
        self.cache[(i, j)] = term
        return term

    def compute(
        self, points: Type.POINT, size: Type.SIZE, direction: Type.DIRECTION
    ):
        # TODO refactor to singular point arg name
        self.reset_cache()

        result = self.get_term(
            i=self.order,
            j=self.order,
            point=points,
            size=size,
            direction=direction,
        )

        self.reset_cache()

        return result


methods = {
    method.id: method
    for method in [
        DefaultForward,
        DefaultBackward,
        DefaultCentral,
        DefaultRichardson,
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
            raise ValueError(
                "Do not simultaneously specify `directions` as a dictionary, and direction IDs."
            )
        ids = list(directions.keys())
        directions = list(directions.values())

    if indices is not None and directions is not None:
        raise ValueError(
            "Do not specify indices if directions are specified, as indices will not be used."
        )

    # Get default directions.
    if point is not None:
        if directions is not None:
            raise NotImplementedError(
                "Please supply only one of `point` and `directions`."
            )
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
        ids = [f"direction_{index}" for index in range(len(directions))]

    # Ensure sufficient IDs.
    if len(ids) != len(directions):
        ids = [
            # Old IDs are assumed to be the IDs of the first `len(ids)` indices.
            *ids,
            # Default IDs are generated for the remaining directions.
            *[
                f"direction_{index}"
                for index in range(len(ids), len(directions))
            ],
        ]
    if len(set(ids)) != len(directions):
        raise ValueError(
            "An error occurred related to IDs. Possible clause: duplicate IDs in the supplied `ids`."
        )

    return ids, directions
