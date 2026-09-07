"""Compare fiddy's engine against `numdifftools` and `scipy.differentiate`.

Informational only, not a competitive benchmark -- see `README.md` in
this directory. Both are optional, imported lazily here (install via
`pip install fiddy[benchmark]`); this module is not imported by
`fiddy/*`.

Every engine is compared on **accuracy per function evaluation**: each
benchmark function's callable is wrapped with a call counter shared across
all three engines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fiddy.estimate import estimate_directional_derivative

from .functions import ALL, BenchmarkFunction

__all__ = ["ComparisonResult", "run_all", "print_report"]


@dataclass
class ComparisonResult:
    name: str
    engine: str
    value: float
    relative_error: float
    """``float("nan")`` where the benchmark function has no true derivative
    (kink/discontinuity) -- accuracy cannot be judged there, only whether
    the engine flagged it as unreliable (see `status`)."""
    n_calls: int
    status: str


def _relative_error(value: float, true_value: float) -> float:
    if math.isnan(true_value):
        return float("nan")
    return abs(value - true_value) / max(abs(true_value), 1e-300)


def _counting_wrapper(function):
    calls = [0]

    def wrapped(point):
        calls[0] += 1
        return function(point)

    return wrapped, calls


def run_fiddy(bf: BenchmarkFunction) -> ComparisonResult:
    wrapped, calls = _counting_wrapper(bf.function)
    result = estimate_directional_derivative(wrapped, bf.point, bf.direction)
    return ComparisonResult(
        bf.name,
        "fiddy",
        result.value,
        _relative_error(result.value, bf.true_derivative),
        calls[0],
        result.status,
    )


def run_numdifftools(bf: BenchmarkFunction) -> ComparisonResult | None:
    try:
        import numdifftools as nd
    except ImportError:
        return None

    def scalar_f(t):
        return float(bf.function(bf.point + t * bf.direction).reshape(-1)[0])

    wrapped, calls = _counting_wrapper(scalar_f)
    value = float(nd.Derivative(wrapped)(0.0))
    return ComparisonResult(
        bf.name,
        "numdifftools",
        value,
        _relative_error(value, bf.true_derivative),
        calls[0],
        "n/a",
    )


def run_scipy(bf: BenchmarkFunction) -> ComparisonResult | None:
    try:
        from scipy.differentiate import derivative
    except ImportError:
        return None

    def vector_f(t):
        # scipy.differentiate.derivative passes an array of points and
        # requires the output shape to match it exactly.
        t_arr = np.asarray(t)
        flat = t_arr.ravel()
        values = np.array(
            [
                bf.function(bf.point + ti * bf.direction).reshape(-1)[0]
                for ti in flat
            ]
        )
        return values.reshape(t_arr.shape)

    wrapped, calls = _counting_wrapper(vector_f)
    res = derivative(wrapped, 0.0)
    value = float(res.df)
    status = "n/a" if bool(res.success) else "failed"
    return ComparisonResult(
        bf.name,
        "scipy.differentiate",
        value,
        _relative_error(value, bf.true_derivative),
        calls[0],
        status,
    )


def run_all(
    functions: list[BenchmarkFunction] | None = None,
) -> list[ComparisonResult]:
    if functions is None:
        functions = ALL
    results = []
    for bf in functions:
        results.append(run_fiddy(bf))
        for runner in (run_numdifftools, run_scipy):
            result = runner(bf)
            if result is not None:
                results.append(result)
    return results


def print_report(results: list[ComparisonResult]) -> None:
    print("Informational only, not a competitive benchmark -- see README.md.")
    print()
    header = f"{'function':22s} {'engine':20s} {'rel_error':>12s} {'n_calls':>8s} {'status':>18s}"
    print(header)
    print("-" * len(header))
    for r in results:
        rel_error = (
            "n/a (nan)"
            if math.isnan(r.relative_error)
            else f"{r.relative_error:.3e}"
        )
        print(
            f"{r.name:22s} {r.engine:20s} {rel_error:>12s} {r.n_calls:>8d} {r.status:>18s}"
        )


if __name__ == "__main__":
    print_report(run_all())
