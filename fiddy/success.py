import abc
from typing import Any, Callable, Union

import numpy as np


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
    ):
        super().__init__()
        if computer_parser is None:
            computer_parser = lambda computer, size: computer.value if computer.size == size else None
        self.computer_parser = computer_parser
        if analysis_parser is None:
            analysis_parser = lambda analysis: analysis.value if computer.size == size else None
        self.analysis_parser = analysis_parser

        self.rtol = rtol
        self.atol = atol

    def method(self, directional_derivative: DirectionalDerivative) -> [bool, float]:
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
            success_by_size[size] = np.isclose(values, values[0], rtol=self.rtol, atol=self.atol).all()

        consistent_results = [
            value
            for size, success in success_by_size.items()
            for value in results_by_size[size].values()
            if success
        ]

        success = np.isclose(consistent_results, consistent_results[0], rtol=self.rtol, atol=self.atol).all()

        #success = np.isclose
        value = np.average(consistent_results)

        return success, value
