import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from fiddy.estimate import estimate_directional_derivative  # noqa: E402
from fiddy.noise import estimate_noise_floor  # noqa: E402
from fiddy.plotting import (  # noqa: E402
    plot_discontinuity,
    plot_extrapolation,
    plot_noise_floor,
    plot_step_ladder,
)
from fiddy.step_size import build_step_ladder  # noqa: E402


def test_plot_noise_floor_runs():
    def f(x):
        return np.array([np.sin(x[0])])

    result = estimate_noise_floor(f, np.array([0.7]), np.array([1.0]))
    ax = plot_noise_floor(result)
    assert ax is not None


def test_plot_step_ladder_runs():
    point = np.array([1.0])
    direction = np.array([1.0])
    ladder = build_step_ladder(point, direction, noise_floor=1e-9)
    values = np.sin(point[0] + ladder) - np.sin(point[0] - ladder)
    ax = plot_step_ladder(ladder, values)
    assert ax is not None


def test_plot_extrapolation_runs():
    def f(x):
        return np.array([np.sin(x[0])])

    result = estimate_directional_derivative(
        f, np.array([1.3]), np.array([1.0])
    )
    axes = plot_extrapolation(result.extrapolation)
    assert axes is not None


def test_plot_discontinuity_runs():
    def f(x):
        return np.array([abs(x[0])])

    result = estimate_directional_derivative(
        f, np.array([0.0]), np.array([1.0])
    )
    ax = plot_discontinuity(result.discontinuity)
    assert ax is not None
