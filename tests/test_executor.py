import numpy as np

from fiddy.estimate import estimate_directional_derivative, estimate_gradient
from fiddy.executor import JoblibExecutor, SequentialExecutor


def test_sequential_executor_preserves_order():
    executor = SequentialExecutor()
    results = executor(lambda x: x**2, [1.0, 2.0, 3.0])
    assert results == [1.0, 4.0, 9.0]


def test_joblib_executor_preserves_order():
    executor = JoblibExecutor(n_jobs=2)
    results = executor(lambda x: x**2, [1.0, 2.0, 3.0, 4.0])
    assert results == [1.0, 4.0, 9.0, 16.0]


def test_switching_executor_does_not_change_directional_derivative_result():
    def f(x):
        return np.array([np.sin(x[0]) * np.exp(0.1 * x[0])])

    x0 = np.array([1.3])
    direction = np.array([1.0])

    sequential = estimate_directional_derivative(
        f, x0, direction, executor=SequentialExecutor()
    )
    parallel = estimate_directional_derivative(
        f, x0, direction, executor=JoblibExecutor(n_jobs=2)
    )

    assert sequential.value == parallel.value
    assert sequential.error_estimate == parallel.error_estimate
    assert sequential.status == parallel.status


def test_switching_executor_does_not_change_gradient_result():
    def f(x):
        return np.array([np.sin(x[0]) + np.cos(x[1]) * x[0]])

    point = np.array([1.1, -0.4])

    sequential = estimate_gradient(f, point, executor=SequentialExecutor())
    parallel = estimate_gradient(f, point, executor=JoblibExecutor(n_jobs=2))

    assert len(sequential) == len(parallel) == 2
    for s, p in zip(sequential, parallel, strict=True):
        assert s.value == p.value
        assert s.error_estimate == p.error_estimate
        assert s.status == p.status
