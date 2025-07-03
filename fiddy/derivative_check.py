import abc
from typing import Any, Callable, Dict, List, Union
from itertools import chain
from dataclasses import dataclass
from typing import Any


import numpy as np
import pandas as pd
import math

from .constants import Type
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
    output: dict[str, Any] = None
    """Miscellaneous output from the method."""


@dataclass
class DerivativeCheckResult:
    method_id: str
    """The method that determined whether the directional derivative is correct."""
    directional_derivative_check_results: list[
        DirectionalDerivativeCheckResult
    ]
    """The results from checking individual directions."""
    test: Type.DERIVATIVE
    """The value that was tested."""
    expectation: Type.DERIVATIVE
    """The expected value."""
    success: bool
    """Whether the check passed."""
    output: dict[str, Any] = None
    """Miscellaneous output from the method."""

    @property
    def df(self):
        df = pd.DataFrame(self.directional_derivative_check_results)
        # FIXME string literal
        df.set_index("direction_id", inplace=True)
        df["abs_diff"] = np.abs(df["expectation"] - df["test"])
        df["rel_diff"] = df["abs_diff"] / np.abs(df["expectation"])
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

        self.output_indices = self.expectation.shape[: -len(self.point.shape)]

    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)

    @abc.abstractmethod
    def method(self, *args, **kwargs):
        raise NotImplementedError


class NumpyIsCloseDerivativeCheck(DerivativeCheck):
    method_id = "np.isclose"

    def method(self, *args, **kwargs):
        directional_derivative_check_results = []
        expected_values, test_values = get_expected_and_test_values(
            self.derivative.directional_derivatives
        ):
            test_value = np.asarray(directional_derivative.value)

            expected_value = []
            for output_index in np.ndindex(self.output_indices):
                element = self.expectation[output_index][direction_index]
                expected_value.append(element)
            expected_value = np.array(expected_value).reshape(test_value.shape)

        for (
            direction_index,
            directional_derivative,
            expected_value,
            test_value,
        ) in enumerate(zip(
            self.derivative.directional_derivatives,
            expected_values,
            test_values,
        )):
            test_result = np.isclose(
                test_value,
                expected_value,
                *args,
                **kwargs,
            )
            directional_derivative_check_result = (
                DirectionalDerivativeCheckResult(
                    direction_id=directional_derivative.id,
                    method_id=self.method_id,
                    test=test_value,
                    expectation=expected_value,
                    output={"return": test_result},
                    success=test_result.all(),
                )
            )

            directional_derivative_check_results.append(
                directional_derivative_check_result
            )

        success = all(r.success for r in directional_derivative_check_results)
        derivative_check_result = DerivativeCheckResult(
            method_id=self.method_id,
            directional_derivative_check_results=directional_derivative_check_results,
            test=self.derivative.value,
            expectation=self.expectation,
            success=success,
        )
        return derivative_check_result


def get_expected_and_test_values(directional_derivatives):
    expected_values = []
    test_values = []
    for direction_index, directional_derivative in enumerate(
        directional_derivatives
    ):
        test_value = directional_derivative.value
        test_values.append(test_value)

        expected_value = []
        for output_index in np.ndindex(self.output_indices):
            element = self.expectation[output_index][direction_index]
            expected_value.append(element)
        expected_value = np.array(expected_value).reshape(test_value.shape)
        expected_values.append(expected_value)

    return expected_values, test_values


class HybridDerivativeCheck(DerivativeCheck):
    """HybridDerivativeCheck.

    The method checks, if gradients are in finite differences range [min, max],
    using forward, backward and central finite differences for potential
    multiple stepsizes eps. If true, gradients will be checked for each
    parameter and assessed whether or not gradients are within acceptable
    absolute tolerances.
    .. math::
        \\frac{|\\mu - \\kappa|}{\\lambda} < \\epsilon
    """

    method_id = "hybrid"

    def method(self, *args, **kwargs):
        success = True
        expected_values, test_values = get_expected_and_test_values(
            self.derivative.directional_derivatives
        )

        results_all = []
        directional_derivative_check_results = []
        for step_size in range(0, len(expected_values)):
            approxs_for_param = []
            grads_for_param = []
            results = []
            for diff_index, directional_derivative in enumerate(
                self.derivative.directional_derivatives
            ):
                try:
                    for grad, approx in zip(
                        expected_values[diff_index - 1][step_size - 1],
                        test_values[diff_index - 1][step_size - 1],
                    ):
                        approxs_for_param.append(approx)
                        grads_for_param.append(grad)
                    fd_range = np.percentile(approxs_for_param, [0, 100])
                    fd_mean = np.mean(approxs_for_param)
                    grad_mean = np.mean(grads_for_param)
                    if not (fd_range[0] <= grad_mean <= fd_range[1]):
                        if np.any(
                            [
                                abs(x - y) > kwargs["atol"]
                                for i, x in enumerate(approxs_for_param)
                                for j, y in enumerate(approxs_for_param)
                                if i != j
                            ]
                        ):
                            fd_range = abs(fd_range[1] - fd_range[0])
                            if (
                                abs(grad_mean - fd_mean)
                                / abs(fd_range + np.finfo(float).eps)
                            ) > kwargs["rtol"]:
                                results.append(False)
                            else:
                                results.append(True)
                        else:
                            results.append(
                                None
                            )  # can't judge consistency / questionable grad approxs
                    else:
                        fd_range = abs(fd_range[1] - fd_range[0])
                        if not np.isfinite([fd_range, fd_mean]).all():
                            results.append(None)
                        else:
                            result = True
                        results.append(result)
                except (IndexError, TypeError) as err:
                    raise ValueError(
                        f"Unexpected error encountered (This should never happen!)"
                    ) from err

                directional_derivative_check_result = (
                    DirectionalDerivativeCheckResult(
                        direction_id=directional_derivative.id,
                        method_id=self.method_id,
                        test=test_value,
                        expectation=expected_value,
                        output={"return": results},
                        success=all(results),
                    )
                )
                directional_derivative_check_results.append(
                    directional_derivative_check_result
                )
                results_all.append(results)

        success = all(chain(*results_all))
        derivative_check_result = DerivativeCheckResult(
            method_id=self.method_id,
            directional_derivative_check_results=directional_derivative_check_results,
            test=self.derivative.value,
            expectation=self.expectation,
            success=success,
        )
        return derivative_check_result
