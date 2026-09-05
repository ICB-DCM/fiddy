"""Optional diagnostic plotting utilities.

These plots exist primarily as a *debugging* aid for understanding the
noise-floor/step-size/extrapolation engine, not only as an end-user
nicety -- a real false-convergence bug (see :mod:`fiddy.extrapolation`'s
module docstring) was only found by printing raw arrays and reasoning
about them by hand; :func:`plot_extrapolation` below would have caught it
on sight.

Requires ``matplotlib``, which is an optional dependency (``pip install
fiddy[examples]`` or plain ``pip install matplotlib``) -- never a core
runtime dependency of fiddy, so it is imported lazily here rather than at
module load time.
"""

from __future__ import annotations

import numpy as np

from .discontinuity import DiscontinuityCheck
from .extrapolation import ExtrapolationResult
from .noise import NoiseFloor

__all__ = [
    "plot_noise_floor",
    "plot_step_ladder",
    "plot_extrapolation",
    "plot_discontinuity",
]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "fiddy.plotting requires matplotlib. Install it with "
            "`pip install matplotlib` or `pip install fiddy[examples]`."
        ) from error
    return plt


def plot_noise_floor(noise_floor: NoiseFloor, ax=None):
    """Plot the per-order noise estimate vs. finite-difference order.

    Makes directly visible the "rapid shrinkage, then plateau" pattern
    :func:`fiddy.noise.estimate_noise_floor` looks for, including which
    order it picked and why (the plateau, if any, is where the curve goes
    flat).

    :param noise_floor: A result from :func:`fiddy.noise.estimate_noise_floor`
        or :func:`fiddy.noise.estimate_model_noise_floor`.
    :param ax: An existing matplotlib axes to draw on; a new figure/axes
        is created if not given.
    :return: The axes drawn on.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    levels = np.arange(1, len(noise_floor.sigmas) + 1)
    sigmas = [s if s is not None else np.nan for s in noise_floor.sigmas]
    ax.semilogy(levels, sigmas, "o-", label="sigma_level(k)")
    if noise_floor.level is not None:
        ax.axvline(
            noise_floor.level,
            color="gray",
            linestyle="--",
            label=f"detected level ({noise_floor.level})",
        )
    ax.axhline(
        noise_floor.sigma,
        color="gray",
        linestyle=":",
        label=f"noise floor ({noise_floor.sigma:.2e})",
    )
    ax.set_xlabel("finite-difference order k")
    ax.set_ylabel("sigma_level(k)")
    title = "Noise-floor plateau detection"
    if not noise_floor.confident:
        title += " (low confidence -- no plateau found)"
    ax.set_title(title)
    ax.legend()
    return ax


def plot_step_ladder(ladder: np.ndarray, values: np.ndarray, ax=None):
    """Plot raw central-difference estimates across the step-size ladder.

    A precursor to the full step-size "V-curve" (estimated error vs. step
    size): this only shows the ladder and the raw per-rung estimates, not
    a per-rung error estimate -- see :func:`plot_extrapolation` for the
    latter, once an extrapolation table is available.

    :param ladder: Step sizes, e.g. from
        :func:`fiddy.step_size.build_step_ladder`.
    :param values: The central-difference estimate at each step size in
        `ladder`.
    :param ax: An existing matplotlib axes to draw on; a new figure/axes
        is created if not given.
    :return: The axes drawn on.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    ax.semilogx(ladder, values, "o-")
    ax.set_xlabel("step size h")
    ax.set_ylabel("central-difference estimate D(h)")
    ax.set_title("Step-size ladder")
    ax.invert_xaxis()
    return ax


def plot_extrapolation(extrapolation: ExtrapolationResult, axes=None):
    """Plot a Neville-table extrapolation's convergence.

    Two panels: the full-ladder diagonal (increasingly higher-order
    estimates) alongside the two independently-corroborating chains'
    final values, and the successive-difference error sequence alongside
    the reported (corroboration-aware) error estimate. This is the exact
    plot that would catch a false-convergence bug like the one described
    in :mod:`fiddy.extrapolation`'s module docstring on sight: a non-
    monotonic, noisy error sequence and/or a visible gap between the two
    chains' final values, instead of a clean decreasing error curve with
    both chains agreeing.

    :param extrapolation: A result from
        :func:`fiddy.extrapolation.extrapolate_central_differences`.
    :param axes: A pair of existing matplotlib axes
        ``(value_ax, error_ax)`` to draw on; a new figure/axes pair is
        created if not given.
    :return: The axes drawn on.
    """
    plt = _require_matplotlib()
    if axes is None:
        _, axes = plt.subplots(2, 1, sharex=True)
    ax_value, ax_error = axes

    orders = np.arange(1, len(extrapolation.diagonal) + 1)
    ax_value.plot(
        orders, extrapolation.diagonal, "o-", label="full-ladder diagonal"
    )
    ax_value.axhline(
        extrapolation.chain_a_value,
        color="tab:orange",
        linestyle="--",
        label=f"chain A final ({extrapolation.chain_a_value:.4g})",
    )
    ax_value.axhline(
        extrapolation.chain_b_value,
        color="tab:green",
        linestyle="--",
        label=f"chain B final ({extrapolation.chain_b_value:.4g})",
    )
    ax_value.set_ylabel("extrapolated value")
    ax_value.set_title(
        f"Extrapolation convergence (chain disagreement={extrapolation.disagreement:.2e})"
    )
    ax_value.legend()

    errors = np.abs(np.diff(extrapolation.diagonal))
    ax_error.semilogy(
        orders[1:], errors, "o-", label="|successive difference|"
    )
    ax_error.axhline(
        extrapolation.error_estimate,
        color="gray",
        linestyle=":",
        label=f"reported error estimate ({extrapolation.error_estimate:.2e})",
    )
    ax_error.set_xlabel("extrapolation order")
    ax_error.set_ylabel("error")
    ax_error.legend()
    return axes


def plot_discontinuity(discontinuity: DiscontinuityCheck, ax=None):
    """Plot the gap curve used by the kink/discontinuity check.

    ``[f(x+h) - 2f(x) + f(x-h)] / h`` vs. step size h, log-log: for a
    smooth function this gap is `O(h)` and slopes down as h shrinks (a
    reference line at that slope is drawn through `reference_index`'s
    observed value, the same point the actual check's prediction is
    anchored at -- see its module docstring for why that's the vetted
    step's immediate coarser neighbor, not the ladder's widest rung); for
    a genuine kink it stays flat (`O(1)`) regardless of h. This directly
    visualizes why central differences alone miss a kink like `abs(x)` at
    its corner (they'd show a flat line at 0 on a linear plot, giving no
    hint anything is wrong) and what the residual check in
    :func:`fiddy.discontinuity.check_discontinuity` is comparing.

    :param discontinuity: A result from
        :func:`fiddy.discontinuity.check_discontinuity`.
    :param ax: An existing matplotlib axes to draw on; a new figure/axes
        is created if not given.
    :return: The axes drawn on.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()

    ladder = discontinuity.ladder
    gap = np.abs(discontinuity.gap_values)
    ax.loglog(ladder, gap, "o-", label="|gap(h)|")

    # Reference: what a smooth function's gap (O(h)) would look like,
    # anchored at the same point the check's own prediction is.
    ref_idx = discontinuity.reference_index
    reference = gap[ref_idx] * (ladder / ladder[ref_idx])
    ax.loglog(
        ladder, reference, "--", color="gray", label="O(h) reference (smooth)"
    )

    ax.axvline(
        ladder[discontinuity.best_index],
        color="tab:red",
        linestyle=":",
        label="h_b (vetted by extrapolation)",
    )
    ax.set_xlabel("step size h")
    ax.set_ylabel("|gap(h)|")
    title = "Kink/discontinuity check"
    if discontinuity.suspected:
        title += " (discontinuity suspected)"
    ax.set_title(title)
    ax.invert_xaxis()
    ax.legend()
    return ax
