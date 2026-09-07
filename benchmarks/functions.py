"""Synthetic benchmark functions for fiddy's dev-time validation suite.

Generic, in-repo synthetic test functions covering the classes of
behavior the engine must handle robustly (smooth / injected-noise /
discrete / near-zero-gradient / ill-conditioned), each with a known
analytic directional derivative (or ``float("nan")`` where none exists)
for comparison. These are deliberately small, dependency-free stand-ins
for real-model failure modes (e.g. ``near_zero_gradient`` is a synthetic
analogue of a real case where a model's true derivative at one parameter
was comparable in magnitude to the model's own noise floor -- see
:mod:`fiddy.extrapolation`'s module docstring for the full story).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BenchmarkFunction",
    "ALL",
    "smooth",
    "noisy",
    "kink",
    "step_discontinuity",
    "near_zero_gradient",
    "ill_conditioned",
]


@dataclass
class BenchmarkFunction:
    name: str
    category: str
    function: Callable[[np.ndarray], np.ndarray]
    point: np.ndarray
    direction: np.ndarray
    true_derivative: float
    """``float("nan")`` where no true derivative exists (kink/discontinuity)."""
    description: str


def _deterministic_noise(x: float, amplitude: float) -> float:
    """Reproducible pseudo-noise: a stand-in for solver-tolerance-induced
    noise that is still exactly reproducible across repeated evaluations
    at the same point, since it is derived from the input's own bit
    pattern rather than global RNG state.
    """
    bits = np.array([x], dtype=">f8").view(">u8")[0]
    rng = np.random.default_rng(bits)
    return float(rng.uniform(-1, 1)) * amplitude


def smooth() -> BenchmarkFunction:
    def f(x):
        return np.array([np.sin(x[0]) * np.exp(0.1 * x[0])])

    x0 = 1.3
    true = math.cos(x0) * math.exp(0.1 * x0) + math.sin(x0) * 0.1 * math.exp(
        0.1 * x0
    )
    return BenchmarkFunction(
        "smooth",
        "smooth",
        f,
        np.array([x0]),
        np.array([1.0]),
        true,
        "sin(x) * exp(0.1x), no injected noise",
    )


def noisy(amplitude: float = 1e-8) -> BenchmarkFunction:
    def f(x):
        return np.array([np.sin(x[0]) + _deterministic_noise(x[0], amplitude)])

    x0 = 0.7
    return BenchmarkFunction(
        "noisy",
        "noisy",
        f,
        np.array([x0]),
        np.array([1.0]),
        math.cos(x0),
        f"sin(x) + deterministic noise, amplitude={amplitude:.0e}",
    )


def kink() -> BenchmarkFunction:
    def f(x):
        return np.array([abs(x[0])])

    return BenchmarkFunction(
        "kink",
        "discrete",
        f,
        np.array([0.0]),
        np.array([1.0]),
        float("nan"),
        "abs(x) at its corner -- central differences alone are exactly 0",
    )


def step_discontinuity() -> BenchmarkFunction:
    def f(x):
        return np.array([1.0 if x[0] >= 0 else 0.0])

    return BenchmarkFunction(
        "step_discontinuity",
        "discrete",
        f,
        np.array([0.0]),
        np.array([1.0]),
        float("nan"),
        "a genuine jump in the function value itself",
    )


def near_zero_gradient(amplitude: float = 1e-8) -> BenchmarkFunction:
    def f(x):
        return np.array(
            [amplitude * np.sin(x[0]) + _deterministic_noise(x[0], amplitude)]
        )

    x0 = 0.4
    return BenchmarkFunction(
        "near_zero_gradient",
        "near_zero_gradient",
        f,
        np.array([x0]),
        np.array([1.0]),
        amplitude * math.cos(x0),
        "derivative magnitude comparable to the noise floor -- a "
        "synthetic analogue of a real case where this happened on a "
        "genuine ODE-based likelihood (see fiddy.extrapolation's module "
        "docstring)",
    )


def ill_conditioned() -> BenchmarkFunction:
    # Rosenbrock function (Rosenbrock1960 in doc/references.bib): curvature
    # along x0 is ~100x that along x1 near this point, the classic
    # synthetic stand-in for badly scaled problems.
    def f(x):
        return np.array([100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2])

    x0 = np.array([1.2, 1.5])
    direction = np.array([1.0, 0.0])
    true = -400 * x0[0] * (x0[1] - x0[0] ** 2) - 2 * (1 - x0[0])
    return BenchmarkFunction(
        "ill_conditioned",
        "ill_conditioned",
        f,
        x0,
        direction,
        true,
        "Rosenbrock function, d/dx0 direction (curvature scales ~100x "
        "differently between directions near this point)",
    )


ALL: list[BenchmarkFunction] = [
    smooth(),
    noisy(),
    kink(),
    step_discontinuity(),
    near_zero_gradient(),
    ill_conditioned(),
]
