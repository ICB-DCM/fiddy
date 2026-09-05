# Tests

One test file per `fiddy/*.py` module (`test_noise.py` <-> `fiddy/noise.py`,
`test_extrapolation.py` <-> `fiddy/extrapolation.py`, etc.), so a module's
behavior and its tests always live in one obvious place. `test_function.py`
covers both `Function` and `CachedFunction` (both defined in
`fiddy/function.py`); `test_estimate.py`/`test_check.py` cover both the
single-output and multi-output (Jacobian) entry points of
`fiddy/estimate.py`/`fiddy/check.py`, since they share most of their
underlying machinery.

- `helpers.py` -- shared test helpers, importable as `from helpers import
  ...` (plain sibling module; `tests/` has no `__init__.py`, so pytest's
  default import mode already puts this directory on `sys.path`).
  Currently just `deterministic_noise`, a reproducible stand-in for
  per-evaluation noise used across many noise/extrapolation/discontinuity
  tests.
- `test_plotting.py` guards its imports with `pytest.importorskip` for
  optional dependencies (`matplotlib`) that aren't required for the core
  package.

Real-model (not synthetic) validation against an actual AMICI/CVODES
model is a separate, external concern from this synthetic suite -- it
has caught real bugs synthetic tests here missed.
