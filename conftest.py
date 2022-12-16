"""
Handler of local plugins.

A plugin is simply a python file within a python module.

The plugins with the `tests.fixtures` contain thematically grouped fixtures:

- datasets -> all fixtures creating datasets to be used in testing.
- frozen_tests (experimental) -> contains anything related to the checking
  of lens runs against pre compiled results. Feature is non active ATM.
- lens_artifacts -> contains logic wrapping models/data in credo artifacts.
- lens_inits -> contains the lens initialized instance.
"""

pytest_plugins = [
    "tests.fixtures.datasets",
    "tests.fixtures.frozen_tests",
    "tests.fixtures.lens_artifacts",
    "tests.fixtures.lens_inits",
]
