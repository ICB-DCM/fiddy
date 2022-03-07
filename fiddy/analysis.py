import abc
from typing import Any, List, Dict
from dataclasses import dataclass, field


from .constants import Type, MethodId
from .derivative import DirectionalDerivative


@dataclass
class AnalysisResult:
    method_id: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class Analysis:
    only_at_completion: bool = False
    def __init__(self, method: Type.ANALYSIS_METHOD = None):
        if method is not None:
            self.method = method
        self.value = None
        self.results = []

    def __call__(self, directional_derivative: DirectionalDerivative):
        self.value = self.method(directional_derivative=directional_derivative)

    @abc.abstractmethod
    def method(self, directional_derivative: DirectionalDerivative) -> Any:
        raise NotImplementedError


class ApproximateCentral(Analysis):
    """Uses the first valid forward and backward directional derivative computers."""
    # FIXME string literal
    id = 'approximate_central'
    only_at_completion: bool = True
    def __init__(self):
        super().__init__()

    #@staticmethod
    #def get_computer(directional_derivative: DirectionalDerivative, size: Type.SIZE, method: MethodId) -> DirectionalDerivative:
    #    for computer in directional_derivative.computers:
    #        if computer.metadata['size'] == size and computer.method == MethodId.FORWARD:
    #            return computer
    #    return None

    def method(self, directional_derivative: DirectionalDerivative) -> List[Type.DIRECTIONAL_DERIVATIVE]:
        computer_results = directional_derivative.get_computer_results()

        results = {
            MethodId.FORWARD: {},
            MethodId.BACKWARD: {},
        }
        for result in computer_results:
            method_id = result.method_id
            size = result.metadata.get('size', None)
            if size is None:
                continue
            # Skip other method computers.
            if method_id not in results:
                continue
            if size in results[method_id]:
                raise ValueError('Apparently duplicated results: multiple results with the same method and step size. Perhaps the requested sizes contained duplicates, or computer or analysis methods produced duplicate results.')
            results[method_id][size] = result.value

        sizes = set(results[MethodId.FORWARD]).intersection(results[MethodId.BACKWARD])

        for size in sizes:
            forward = results[MethodId.FORWARD][size]
            backward = results[MethodId.BACKWARD][size]
            central = (forward + backward)/2
            result = AnalysisResult(
                method_id=self.id,
                value=central,
                # FIXME string literal
                metadata={'size': size}
            )
            self.results.append(result)
