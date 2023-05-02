import abc
from typing import Any, Callable, Dict, List, Union
import warnings

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import (
    MethodId,
    Type,
    EPSILON,
)

from .analysis import Analysis
from .directional_derivative import methods, get_directions, Computer, DirectionalDerivative

from .success import Success


#@dataclass
#class Analysis:
#    # Change to callable?
#    method: Type.ANALYSIS_METHOD
#    result: Any
#
#    def __call__(self, directional_derivative):
#        self.method(directional_derivative=directional_derivative)


class Derivative:
    # TODO support higher order derivatives (currently only gradient)
    """Handle all aspects of derivative computation.

    The general sequence is:
    1. define derivatives to be computed
    2. compute derivatives (possibly with multiple methods)
    3. analyze derivatives (e.g. compute "consistency" between multiple methods)
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
    relative_sizes: bool = False,
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
        relative_sizes:
            If `True`, sizes are scaled by the `point`, otherwise not.
    """
    # TODO docs
    if directions is not None:
        direction_ids, directions = get_directions(
            directions=directions,
            ids=direction_ids,
            indices=direction_indices,
        )
    else:
        direction_ids, directions = get_directions(
            point=point,
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
                    relative_size=relative_sizes,
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

    return Derivative(
        directional_derivatives=directional_derivatives,
        autorun=True,
        #function=function,
        #request=request,
        #*args,
        #**kwargs,
    )
