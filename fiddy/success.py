import abc
from typing import Any, Callable, Union, Tuple

import numpy as np
import math

from .constants import Type
#from .derivative import Derivative
from .directional_derivative import DirectionalDerivative


class Success:
    id: str = None
    def __init__(self, method: Type.SUCCESS_CHECKER = None):
        if method is not None:
            self.method = method
        self.value = None
        self.success = None
        if self.id is None:
            raise ValueError('Please set the success checker ID in its class.')

    def __call__(self, directional_derivative: DirectionalDerivative):
        self.success, self.value = self.method(directional_derivative=directional_derivative)
        return self.success, self.value

    @abc.abstractmethod
    def method(self, directional_derivative: DirectionalDerivative) -> Any:
        raise NotImplementedError


class Consistency(Success):
    # FIXME string literal
    id = 'consistency'
    only_at_completion: bool = True
    def __init__(
        self,
        computer_parser: Callable[["Computer"], Union[float, None]] = None,
        analysis_parser: Callable[["Analysis"], Union[float, None]] = None,
        rtol: float = 0.2,
        atol: float = 1e-15,
        equal_nan: bool = True,
    ):
        super().__init__()
        if computer_parser is None:
            computer_parser = lambda computer, size: computer.value if computer.size == size else None
        self.computer_parser = computer_parser
        if analysis_parser is None:
            analysis_parser = lambda computer, analysis: analysis.value if computer.size == size else None
        self.analysis_parser = analysis_parser

        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan

    def method(self, directional_derivative: DirectionalDerivative) -> Tuple[bool, float]:
        # FIXME string literals
        computer_results = directional_derivative.get_computer_results()
        analysis_results = directional_derivative.get_analysis_results()
        results_by_size = {}
        for result in [*computer_results, *analysis_results]:
            size = result.metadata.get('size', None)
            if size is None:
                continue
            if size not in results_by_size:
                results_by_size[size] = {}
            if result.method_id in results_by_size[size]:
                raise ValueError(f'Duplicate, and possibly conflicting, results for method "{result.method_id}" and size "{size}".')
            results_by_size[size][result.method_id] = result.value

        success_by_size = {}
        for size, results in results_by_size.items():
            values = list(results.values())
            success_by_size[size] = np.isclose(values, values[0], rtol=self.rtol, atol=self.atol, equal_nan=self.equal_nan).all()

        consistent_results = [
            value
            for size, success in success_by_size.items()
            for value in results_by_size[size].values()
            if success
        ]

        success = False
        if consistent_results:
            success = np.isclose(consistent_results, consistent_results[0], rtol=self.rtol, atol=self.atol, equal_nan=self.equal_nan).all()
        value = np.average(np.array(consistent_results), axis=0)

        return success, value


class ConsistencyNew(Success):
    """ Add documentation here """
    # FIXME: Add documentation
    # FIXME string literal
    id = 'consistencynew'
    only_at_completion: bool = True
    def __init__(
        self,
        computer_parser: Callable[["Computer"], Union[float, None]] = None,
        analysis_parser: Callable[["Analysis"], Union[float, None]] = None,
        rtol: float = 0.2,
        atol: float = 1e-15,
        equal_nan: bool = True,
    ):
        super().__init__()
        if computer_parser is None:
            computer_parser = lambda computer, size: computer.value if computer.size == size else None
        self.computer_parser = computer_parser
        if analysis_parser is None:
            analysis_parser = lambda computer, analysis: analysis.value if computer.size == size else None
        self.analysis_parser = analysis_parser

        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan

    def method(self, directional_derivative: DirectionalDerivative) -> Tuple[bool, float]:
        # FIXME string literals
        computer_results = directional_derivative.get_computer_results()
        analysis_results = directional_derivative.get_analysis_results()
        results_by_size = {}
        for result in [*computer_results, *analysis_results]:
            size = result.metadata.get('size', None)
            if size is None:
                continue
            if size not in results_by_size:
                results_by_size[size] = {}
            if result.method_id in results_by_size[size]:
                raise ValueError(f'Duplicate, and possibly conflicting, results for method "{result.method_id}" and size "{size}".')
            results_by_size[size][result.method_id] = result.value

        success_by_size = {}
        for size, results in results_by_size.items():
            fds = list(results.values()) # values for all methods encountered (FD forward, backward and central (possibly calculated through analysis))
            fd_range = np.percentile(fds, [0, 100]) # range of fds
            fd_mean = np.mean(fds) # calculate mean of fd approximations
            grad = results['approximate_central'] # FIXME how to get actual gradient
            abs_tol = self.atol
            rel_tol = self.rtol
            if not (fd_range[0] <= grad <= fd_range[1]):
                if any(
                    [abs(x-y) > abs_tol for i, x in enumerate(fds)
                        for j, y in enumerate(fds) if i != j]):
                    fd_range = abs(fd_range[1] - fd_range[0])
                    if (((abs(grad - fd_mean) / abs(
                     fd_range+np.finfo(float).eps)) > rel_tol)):
                        success_by_size[size] = False
                    else:
                        success_by_size[size] = True
                else:
                    success_by_size[size] = None
            else:
                fd_range = abs(fd_range[1] - fd_range[0])
                if math.isinf((fd_range) or math.isnan(fd_range)
                   or math.isinf(fd_mean) or math.isnan(fd_mean)):
                    success_by_size[size] = None
                else:
                    success_by_size[size] = True

        consistent_results = [
            value
            for size, success in success_by_size.items()
            for value in results_by_size[size].values()
            if success
        ]

        success = False
        if consistent_results:
            success = np.isclose(consistent_results, consistent_results[0], rtol=self.rtol, atol=self.atol, equal_nan=self.equal_nan).all()
        value = np.average(np.array(consistent_results), axis=0)

        return success, value
