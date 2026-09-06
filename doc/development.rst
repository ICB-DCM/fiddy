Development
===========

Contributing to fiddy
---------------------

We welcome contributions from the community.
If you're interested in contributing to the project, please follow these steps:

1. Fork the repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear messages.
4. Push your changes to your forked repository.
5. Open a pull request against the main repository.

Before submitting a pull request, please ensure that your code adheres to the
project's coding standards and includes appropriate tests.


Development setup
-----------------

We use `pre-commit <https://pre-commit.com/>`_ to run linters and formatters on
the codebase. To enable pre-commit hooks in your development environment, run:

.. code-block:: bash

    pip install pre-commit
    pre-commit install

Python compatibility
--------------------

fiddy follows `NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_.

Running tests
-------------

We use `pytest <https://docs.pytest.org/en/stable/>`_ for testing.

To run the test suite, execute the following command in the project root
directory:

.. code-block:: bash

    pytest

Validating against AMICI's real-model test suites
---------------------------------------------------

fiddy is framework-agnostic by design: nothing in ``fiddy/*`` may import
or reference AMICI, PEtab, or any other specific downstream consumer.
That said, two of AMICI's own test suites are fiddy's primary real-world
robustness indicators, checking fiddy-computed finite-difference
gradients against AMICI's analytic sensitivities: its PEtab
benchmark-collection gradient-check suite
(``tests/benchmark_models/test_petab_benchmark.py``) across dozens of
real ODE models, and its `SBML Test Suite
<https://github.com/sbmlteam/sbml-test-suite>`_ semantic-case suite
(``tests/sbml/testSBMLSuite.py``) across ~1780 small, targeted models --
checking a model's full bundled state/observable/likelihood
sensitivities together via ``check_jacobian``. The latter is a different
real-world signal than the former: it has repeatedly surfaced
discontinuity/event-triggered edge cases (e.g. SBML ``event`` triggers)
that the handful of larger PEtab benchmark models don't happen to
exercise. The steps below let any contributor reproduce either
validation locally.

1. Clone AMICI:

   .. code-block:: bash

       git clone https://github.com/AMICI-dev/AMICI.git /path/to/amici

2. Install AMICI editable from that checkout, into the same environment
   fiddy itself is developed in, with the extras either suite needs:
   ``petab``/``test``/``vis`` for the PEtab benchmark suite, plus
   ``jax`` for the SBML suite's JAX-based sensitivity cross-check (run
   on a couple of its cases):

   .. code-block:: bash

       python -m pip install -e "/path/to/amici/python/sdist[petab,test,vis,jax]"

3. Install *this* fiddy checkout editable, last -- so it overrides the
   ``test`` extra's pinned PyPI ``fiddy`` version:

   .. code-block:: bash

       python -m pip install -e /path/to/fiddy/repo

Running the PEtab benchmark test suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

4. Install ``benchmark_models_petab`` (not on PyPI, not part of any
   AMICI extra):

   .. code-block:: bash

       python -m pip install 'git+https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab.git@master#subdirectory=src/python&egg=benchmark_models_petab'

5. Run the PEtab v1 benchmark gradient-check tests from the AMICI
   checkout root. The ``filterwarnings`` override works around one
   upstream petab-version deprecation warning unrelated to fiddy, needed
   since the repo's own ``pytest.ini`` turns all warnings into errors:

   .. code-block:: bash

       cd /path/to/amici
       python -m pytest tests/benchmark_models/test_petab_benchmark.py::test_benchmark_gradient \
         -o "filterwarnings=ignore::DeprecationWarning:petab" -v

   Use ``--collect-only -q`` first to find exact node IDs; parametrize
   order is ``[problem_id-sensitivity_method-scale]``, e.g.
   ``test_benchmark_gradient[Boehm_JProteomeRes2014-forward-scaled]``.

6. Run the PEtab v2 nominal-parameters-likelihood test the same way:

   .. code-block:: bash

       python -m pytest tests/benchmark_models/test_petab_benchmark.py::test_nominal_parameters_llh_v2 \
         -o "filterwarnings=ignore::DeprecationWarning:petab" -v

Running the SBML semantic test suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

4. Clone the SBML Test Suite itself into the AMICI checkout, at the
   fixed path ``testSBMLSuite.py`` expects:

   .. code-block:: bash

       git clone https://github.com/sbmlteam/sbml-test-suite.git \
         /path/to/amici/tests/sbml/sbml-test-suite

5. Run a subset of cases with ``--cases`` (a comma-separated list of
   5-digit test IDs, e.g. ``00026``) -- running the *entire* suite takes
   a long time and isn't usually necessary for a targeted check:

   .. code-block:: bash

       cd /path/to/amici/tests/sbml
       python -m pytest testSBMLSuite.py --cases=00026,00349,00358

   Omit ``--cases`` to run every case in the suite.

A model's first run compiles a C++ extension (~20s); subsequent runs
reuse the cached build. Build artifacts from either suite land under
``amici.get_model_root_dir()``; set ``AMICI_MODELS_ROOT`` to point
elsewhere (e.g. to avoid clashing with a concurrent run using the same
checkout).

Release process
---------------

Releases are managed via GitHub releases.

To create a new release:

1. Go to the "Releases" section of the GitHub repository.
2. Click on "Draft a new release".
3. Fill in the tag version, release title, and description.

   Version & tag: We follow `Semantic Versioning <https://semver.org/>`_.
   The tag should be in the format ``vX.Y.Z`` (e.g., ``v1.0.0``).
   The release title is ``fiddy vX.Y.Z``.
   The package version will be automatically inferred from the tag
   via `setuptools_scm <https://setuptools-scm.readthedocs.io/en/latest/>`__.

   Description: Include a summary of changes, new features, bug fixes,
   and any other relevant information.

4. Publish the release.

   A GitHub Action workflow will automatically build and upload the package to
   PyPI. Ensure that the action completes successfully.
