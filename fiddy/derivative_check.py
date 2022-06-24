import abc
from typing import Any, Callable, Dict, List, Union
from itertools import product

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import (
    #TYPE_DIMENSION,
    #TYPE_FUNCTION,
    #TYPE_SIZE,
    #TYPE_OUTPUT,
    #TYPE_GRADIENT_FUNCTION,
    MethodId,
    #AnalysisMethod,
    Type,
)

from .directional_derivative import methods, get_directions, Computer, DirectionalDerivative

from .step import step
from .success import Success

from .derivative import Derivative


@dataclass
class DirectionalDerivativeCheckResult:
    direction_id: str
    """The direction."""
    method_id: str
    """The method that determined whether the directional derivative is correct."""
    test: Type.DIRECTIONAL_DERIVATIVE
    """The value that was tested."""
    expectation: Type.DIRECTIONAL_DERIVATIVE
    """The expected value."""
    success: bool
    """Whether the check passed."""
    output: Dict[str, Any] = None
    """Miscellaneous output from the method."""


@dataclass
class DerivativeCheckResult:
    method_id: str
    """The method that determined whether the directional derivative is correct."""
    directional_derivative_check_results: List[DirectionalDerivativeCheckResult]
    """The results from checking individual directions."""
    test: Type.DERIVATIVE
    """The value that was tested."""
    expectation: Type.DERIVATIVE
    """The expected value."""
    success: bool
    """Whether the check passed."""
    output: Dict[str, Any] = None
    """Miscellaneous output from the method."""

    @property
    def df(self):
        df = pd.DataFrame(self.directional_derivative_check_results)
        # FIXME string literal
        df.set_index('direction_id', inplace=True)
        return df


class DerivativeCheck(abc.ABC):
    """Check whether a derivative is correct.

    Args:
        derivative:
            The test derivative.
        expectation:
            The expected derivative.
        point:
            The point where the test derivative was computed.
        output_indices:
            The derivative can be a multi-dimensional object that has dimensions
            associated with the multiple outputs of a function, and dimensions
            associated with the derivative of these multiple outputs with respect
            to multiple directions.
    """
    method_id: str
    """The name of the derivative check method."""

    def __init__(
        self,
        derivative: Derivative,
        expectation: Type.DERIVATIVE,
        point: Type.POINT,
    ):
        self.derivative = derivative
        self.expectation = expectation
        self.point = point

        self.output_indices = self.expectation.shape[:-len(self.point.shape)]

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)

    @abc.abstractmethod
    def method(self, *args, **kwargs):
        raise NotImplementedError


class NumpyIsCloseDerivativeCheck(DerivativeCheck):
    method_id = 'np.isclose'

    def method(self, *args, rtol: float = 1e-2, atol: float = 1e-15, **kwargs):
        directional_derivative_check_results = []
        success = True
        for direction_index, directional_derivative in enumerate(self.derivative.directional_derivatives):
            test_value = directional_derivative.value

            expected_value = []
            for output_index in np.ndindex(self.output_indices):
                element = self.expectation[output_index][direction_index]
                expected_value.append(element)
            expected_value = np.array(expected_value).reshape(test_value.shape)

            test_result = np.isclose(
                test_value,
                expected_value,
                *args,
                rtol=rtol,
                atol=atol,
                **kwargs,
            )

            directional_derivative_check_result = DirectionalDerivativeCheckResult(
                direction_id=directional_derivative.id,
                method_id=self.method_id,
                test=test_value,
                expectation=expected_value,
                output={'return': test_result},
                success=test_result.all(),
            )

            directional_derivative_check_results.append(directional_derivative_check_result)

        success = all([r.success for r in directional_derivative_check_results])
        derivative_check_result = DerivativeCheckResult(
            method_id=self.method_id,
            directional_derivative_check_results=directional_derivative_check_results,
            test=self.derivative.value,
            expectation=self.expectation,
            success=success,
        )
        return derivative_check_result

