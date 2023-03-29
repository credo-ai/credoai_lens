"""
Testing protocols for the Lens package.

Testing evaluators behavior when dealing with regression
data/models.
"""

import pytest


from credoai.evaluators import (
    DataEquity,
    DataProfiler,
    ModelEquity,
    ModelFairness,
    Performance,
    DataFairness,
    Deepchecks,
    ModelProfiler,
)

TEST_METRICS = [["r2_score", "ks_score"]]
TEST_METRICS_IDS = ["regression_metric"]


@pytest.mark.parametrize(
    "evaluator",
    [ModelFairness, Performance],
    ids=["Model Fairness", "Performance"],
)
@pytest.mark.parametrize("metrics", TEST_METRICS, ids=TEST_METRICS_IDS)
def test_modelfairness_performance(init_lens_regression, metrics, evaluator):
    lens, temp_file, gov = init_lens_regression
    eval = evaluator(metrics)
    lens.add(eval)
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))


@pytest.mark.parametrize(
    "evaluator",
    [
        DataFairness,
        DataProfiler,
        ModelEquity,
        DataEquity,
        Deepchecks,
        ModelProfiler,
    ],
    ids=[
        "DataFairness",
        "DataProfiler",
        "ModelEquity",
        "DataEquity",
        "Deepchecks",
        "ModelProfiler",
    ],
)
def test_generic_evaluator(init_lens_regression, evaluator):
    """
    Any evaluator not requiring specific treatment can be tested here
    """
    lens, temp_file, gov = init_lens_regression
    lens.add(evaluator())
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))
