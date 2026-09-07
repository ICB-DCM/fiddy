import pytest

pytest.importorskip("numdifftools")
scipy_differentiate = pytest.importorskip("scipy.differentiate")

from benchmarks.compare import run_all  # noqa: E402
from benchmarks.functions import ALL  # noqa: E402


def test_all_benchmark_functions_have_a_direction_and_point():
    for bf in ALL:
        assert bf.point.shape[0] == bf.direction.shape[0]


def test_comparison_harness_runs_for_every_function_and_engine():
    results = run_all()

    # fiddy + numdifftools + scipy.differentiate, for every function.
    assert len(results) == 3 * len(ALL)
    engines = {r.engine for r in results}
    assert engines == {"fiddy", "numdifftools", "scipy.differentiate"}

    for r in results:
        assert r.n_calls > 0


def test_fiddy_flags_unreliable_results_as_such():
    """Unlike numdifftools/scipy.differentiate, fiddy must never report a
    confident value for a direction it cannot actually resolve -- the
    smooth/ill-conditioned cases should converge; kink/discrete/near-zero
    cases must not claim "converged"."""
    results = {r.name: r for r in run_all() if r.engine == "fiddy"}

    assert results["smooth"].status == "converged"
    assert results["ill_conditioned"].status == "converged"
    for name in ("kink", "step_discontinuity", "near_zero_gradient"):
        assert results[name].status != "converged"
