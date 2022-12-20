# How to write tests

The testing protocol of choice is [pytest](https://docs.pytest.org/en/7.2.x/). According to pytest naming conventions, any file containing tests has to be named with the prefix "test_". Within each of the test files, each test function has to be named using the same prefix, while test classes follow the CamelCase convention: "TestMyFunctionName".

## Test files

The `tests` directory contains the following files:

* **Conftest.py**: the general place in which (fixtures)[https://docs.pytest.org/en/6.2.x/fixture.html] are stored.
The fixtures contained in this files are not data/lens specific, but
of a general utility for testing. For the organization of lens specific
fixtures see [the next section](#fixtures)
* **test_binary_classification.py**: `Lens` tests involving binary classification data.
* **test_multiclass_classification.py**: `Lens` tests involving multiclass classification data.
* **test_regression.py**:`Lens` tests involving regression data.
* **test_integration.py**: tests downloading an assessment plan, executing the associated `Lens` run, and exporting results to file and to
platform.
* **test_quickstart.py**: tests the logic in the quick-start notebook.
* **tests_frozen_results.py**: (Experimental) tests results of evaluators by comparing them to stored values. Currently all tests here are skipped.
* **test_artifacts.py**: tests specific functionality of `Lens` artifacts.

## Lens Fixtures
All fixtures inherent to datasets, `Lens` artifacts and `Lens` initialization are thematically organized and contained in the **fixtures** module. The fixtures are separated in the following
files:

- **datasets** -> all fixtures creating datasets to be used in testing.
- **frozen_tests** (experimental) -> contains anything related to the checking
  of lens runs against pre compiled results. Feature is non active ATM.
- **lens_artifacts** -> contains logic wrapping models/data in credo artifacts.
- **lens_inits** -> contains the lens initialized instance.

The files are seen by pytest as plugins. In order to add the files as plugins, their path needs to be added to a **project level** `conftest.py` file. Any fixtures added to these files will be immediately available
for testing, in order to add new plugins to pytest please edit the conftest file in the credo-lens directory.

> **Warning**
> This `conftest.py` file should not be confused with the one inside the > `tests` folder which is used only for general fixtures.


## Evaluator testing

The main structure of an evaluator integration test is the following:

```python
def test_generic_evaluator(init_lens_classification, evaluator):
    """
    Any evaluator not requiring specific treatment can be tested here
    """
    lens, temp_file, gov = init_lens_classification
    lens.add(evaluator())
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))
```
In the code above, `init_lens_classification` is a fixture that contains an initialized `Lens` instance, a reference to a temporary file (this is created using [temp_path](https://credo-ai.atlassian.net/jira/projects), a pytest fixture) necessary for governance testing, and an instantiated governance object.

In order to check multiple assertion in a single test, while still allowing pytest to separate assertion results, we leverage [pytest-assume](https://github.com/astraw38/pytest-assume).

All evaluator tests will mandatorily check, in order:

1. Correct production of results
2. Correct output into evidence format
3. Correct behavior while sending evidence to the governance object
4. Correct behavior while exporting evidences to file.

Any extra check on result values/structure will be appended after the mandatory checks.

### Fixture creation
In general any operation required to create datasets, credoai-lens artifacts, and instantiating `Lens` is wrapped in fixtures. These fixtures are placed in files according to the logic highlighted in 
the section [Fixtures](#fixtures).





