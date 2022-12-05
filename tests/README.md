# How to write tests

The testing protocol of choice is [pytest](https://docs.pytest.org/en/7.2.x/). According to pytest naming conventions, any file containing tests has to be named with the prefix "test_". Within each of the test files, each test function has to be named using the same prefix, while test classes follow the CamelCase convention: "TestMyFunctionName".

## Test files

The `tests` directory contains the following files:

* **Conftest.py**: the general place in which (fixtures)[https://docs.pytest.org/en/6.2.x/fixture.html] are stored.
* **test_lens.py**: contains all integration tests for evaluators, i.e., evaluators are tested within the
  lens framework. Also evidence exporting through governance artifacts is tested for each evaluator
* **test_regression.py**: contains integration tests for evaluators, but run on regression models.
  !! This is  temporary, these tests will be moved to *test_lens.py*.
* **test_quickstart.py**: tests the logic in the quickstart notebook.
* **tests_frozen_results.py**: tests results of evaluators by comparing them to stored values.

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
In general any operation required to create datasets, credoai-lens artifacts, and instantiating `Lens` is wrapped in fixtures. These fixtures are located in the *conftest.py* file.

The structure of *conftest* is formalized within the file itself, by explicitly defined sections.






