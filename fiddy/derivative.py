import abc
from typing import Any, Callable, Dict, List, Union

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import (
    MethodId,
    Type,
)

from .directional_derivative import methods, get_directions, Computer, DirectionalDerivative

from .step import step
from .success import Success



"""
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
    value: Type.DIRECTIONAL_DERIVATIVE = None

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
        self.value = self.method(
            point=self.point,
            direction=self.direction,
            size=self.size,
        )
        self.completed = True
"""


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


@dataclass
class Analysis:
    # Change to callable?
    method: Type.ANALYSIS_METHOD
    result: Any

    def __call__(self, directional_derivative):
        self.method(directional_derivative=directional_derivative)


"""
@dataclass
class DirectionalDerivative:
    # Each gradient result should have a unique method+step combination.
    id: str
    pending_computers: List[Computer]
    computers: List[Computer]
    analyses: List[Analysis]
    # A method that acts on the information in the direction result,
    # including gradient approximation values and post-processing results,
    # to determine whether the gradient was computed successfully.
    # TODO alternatively, can be interpreted as whether the gradient check worked.
    success_method: Callable[["DirectionalDerivative"], bool] = lambda _: True
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
        # TODO decide whether to do this here and/or in `run_next_computer` and `run_success_method`
        self.completed = True

    def iterate(self):
        self.run_next_computer()
        self.run_analyses()
        self.run_success_method()

    def run_next_computer(self):
        try:
            computer = self.pending_computers.pop(0)
        except IndexError:
            self.completed = True
            return
        computer()
        self.computers.append(computer)

    def run_analyses(self):
        for analysis in self.analyses:
            analysis(self)

    def run_success_method(self):
        self.success, value = self.success_method(self)
        if self.success:
            self.value = value
            if self.fast:
                self.completed = True

"""

class Derivative:
    # TODO support higher order derivatives (currently only gradient)
    """Handle all aspects of derivative computation.

    The general sequence is:
    1. define derivatives to be computed
    2. compute derivatives (possibly with multiple different methods)
    3. analyze derivatives (e.g. compute "consistency" between different methods)
    4. check derivatives (e.g. ensure sufficient "consistency")

    Attributes:
        directional_derivatives:
            A list of directional derivative objects.
        expected_derivative:
            The expected derivative.
        analysis_results:
            A list of analysis result objects.
        success_checker:
            The method to determine whether the derivative was successfully computed.
        success:
            Whether the derivative was successfully computed.
    """
    hide_columns = [
        "pending_computers",
        "computers",
        "analyses",
        "success_checker",
        "expected_result",
        "fast",
        "autorun",
    ]


    def __init__(
        self,
        # function: TYPE_FUNCTION,
        directional_derivatives: List[DirectionalDerivative],
        autorun: bool = True,
        # custom_methods: Dict[str, TYPE_FUNCTION] = None,
    ):
        # self.function = function
        # self.request = request

        # self.custom_methods = custom_methods
        # if custom_methods is None:
        #     self.custom_methods = {}

        # if autorun:
        #     # FIXME
        #     pass
        self.directional_derivatives = directional_derivatives

    @property
    def df_full(self):
        _df = pd.DataFrame(data=self.directional_derivatives)
        # FIXME string literal
        _df.set_index('id', inplace=True)
        return _df

    @property
    def series(self):
        # FIXME string literal
        return self.df['value']

    @property
    def dict(self):
        return dict(self.series)

    @property
    def value(self):
        return np.stack(self.series.values, axis=-1)

    @property
    def df(self):
        computer_results = [
            directional_derivative.get_computer_results()
            for directional_derivative in self.directional_derivatives
        ]
        analysis_results = [
            directional_derivative.get_analysis_results()
            for directional_derivative in self.directional_derivatives
        ]
        df = self.df_full.drop(columns=self.hide_columns)
        df['computer_results'] = [pd.DataFrame(data=results) for results in computer_results]
        df['analysis_results'] = [pd.DataFrame(data=results) for results in analysis_results]
        df.index.rename('direction', inplace=True)
        return df

    def print(self):
        print(self.df)


def get_derivative(
    function: Type.FUNCTION,
    point: Type.POINT,
    # TODO store these defaults in some dictionary that a user can easily modify
    # TODO Default to some list e.g. [1e-3, 1e-5]
    sizes: List[Type.SIZE],
    # TODO Default to e.g. ["forward", "backward", "central"]
    method_ids: List[Union[str, MethodId]],
    # TODO for gradient check; add support for methods
    # TODO add some default consistency check
    # TODO change to class that can be initialized with.. directional_derivative object?
    success_checker: Success,
    *args,
    analysis_classes: List[Analysis] = None,
    directions: Union[List[Type.DIRECTION], Dict[str, Type.DIRECTION]] = None,
    direction_ids: List[str] = None,
    direction_indices: List[int] = None,
    custom_methods: Dict[str, Callable] = None,
    expected_result: List[Type.SCALAR] = None,
    **kwargs,
):
    """Get a derivative.

    Args:
        sizes:
            The step sizes.
        direction_ids:
            The IDs of the directions.
        directions:
            List: The directions to step along. Dictionary: keys are direction IDs, values are directions.
    """
    # TODO docs
    direction_ids, directions = get_directions(
        point=point,
        directions=directions,
        ids=direction_ids,
        indices=direction_indices,
    )
    if custom_methods is None:
        custom_methods = {}
    if analysis_classes is None:
        analysis_classes = []
    directional_derivatives = []
    for direction_id, direction in zip(direction_ids, directions):
        computers = []
        for size in sizes:
            for method_id in method_ids:
                method = custom_methods.get(
                    method_id,
                    methods.get(
                        method_id,
                        None,
                    )
                )(function=function)
                if method is None:
                    raise NotImplementedError(
                        f"The requested method '{method_id}' is not a default method. Please supply it as a custom method."
                    )
                computer = Computer(
                    function=function,
                    point=point,
                    direction=direction,
                    size=size,
                    method=method,
                    autorun=False,
                )
                computers.append(computer)
        directional_derivative = DirectionalDerivative(
            id=direction_id,
            direction=direction,
            pending_computers=computers,
            # TODO support users supplying previously run computers?
            computers=[],
            # TODO convert str to method. default?
            analyses=[analysis_class() for analysis_class in analysis_classes],
            # TODO proper `def` default
            success_checker=success_checker if success_checker is not None else lambda _: True,
        )
        directional_derivatives.append(directional_derivative)

    #analysis_results = []
    #for analysis in analyses:
    #    analysis_result = AnalysisResult(
    #        method=analysis,
    #        result=None,
    #    )
    #    analysis_results.append(analysis_result)

    #    #dimension_result = DimensionResult(
    #    #    function=function,
    #    #    gradient_results=gradient_results,
    #    #    expected_result=List[SCALAR],
    #    #    analysis_results=analysis_results,
    #    #    result_method=result_method,
    #    #    result=False,
    #    #    completed=False,
    #    #)
    #    #request.append(dimension_result)
    return Derivative(
        directional_derivatives=directional_derivatives,
        autorun=True,
        #function=function,
        #request=request,
        #*args,
        #**kwargs,
    )
