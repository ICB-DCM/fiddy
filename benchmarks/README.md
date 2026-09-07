# Benchmarks

Generic, in-repo synthetic test functions covering the classes of
behavior fiddy's engine must handle robustly, and a harness comparing
fiddy's engine against `numdifftools` and `scipy.differentiate` on
accuracy-per-evaluation.

**This comparison is informational only -- fiddy is not trying to win on
cost-per-evaluation or raw accuracy against a well-behaved function.**
This suite exists to keep that trade-off honest and visible over time
(e.g. to notice if a future change makes the cost/accuracy trade-off
meaningfully worse), not to chase a leaderboard position.

This is **dev-only tooling, not part of the `fiddy` public API** and not
covered by the package's normal test suite in the same way `tests/` is.
`numdifftools` and a recent-enough `scipy` (for `scipy.differentiate`,
added in SciPy >=1.15) are optional comparison baselines here, not fiddy
runtime dependencies (fiddy's core engine is deliberately self-contained
and numpy-only) -- install them via the `benchmark` extra:
```bash
pip install "fiddy[benchmark]"
```

- `functions.py` — the synthetic test functions (smooth / noisy / discrete
  / near-zero-gradient / ill-conditioned), each with a known analytic
  derivative.
- `compare.py` — runs fiddy, `numdifftools`, and `scipy.differentiate`
  against every function and reports relative error and function-evaluation
  count per engine. Run directly for a printed report:
  `python -m benchmarks.compare`.
