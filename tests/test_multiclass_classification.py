"""
Testing protocols for the Lens package.

Testing evaluators behavior when dealing with multiclass classification
data/models.
"""

import pytest

from credoai.evaluators import (
    ModelFairness,
    Performance,
    DataProfiler,
    ModelProfiler,
    Deepchecks,
    ModelEquity,
    Security,
)
from credoai.modules.constants_metrics import FAIRNESS_FUNCTIONS


##################################################
#################### Init ########################
##################################################

TEST_METRICS = [
    ["false_negative_rate"],
    ["precision_score"],
    ["demographic_parity_difference"],
    ["demographic_parity_ratio"],
    ["equalized_odds_difference"],
    ["equal_opportunity_difference"]
    # ["roc_curve"],
    # ["false_negative_rate", "average_precision_score", "equal_opportunity"],
]
TEST_METRICS_IDS = [
    "binary_metric",
    "precision",
    "demographic_parity_difference",
    "demographic_parity_ratio",
    "equalized_odds_difference",
    "equal_opportunity_difference",
]


@pytest.fixture
def xfail_fairness_tests(request):
    """
    Fail tests for fairness metrics.

    TODO: explore multiclass definitions of fairness and updated the tests
    if any logic change is made.
    """
    message = "Fairness metrics currently not supported for multiclass classification"
    metrics = request.getfixturevalue("metrics")
    evaluator = request.getfixturevalue("evaluator")

    metrics_fairness_intersection = [x for x in metrics if x in FAIRNESS_FUNCTIONS]

    if evaluator == ModelFairness and len(metrics_fairness_intersection) != 0:
        request.node.add_marker(pytest.mark.xfail(reason=message))


##################################################
#################### Tests #######################
##################################################


@pytest.mark.parametrize(
    "evaluator", [ModelFairness, Performance], ids=["Model Fairness", "Performance"]
)
@pytest.mark.parametrize("metrics", TEST_METRICS, ids=TEST_METRICS_IDS)
@pytest.mark.usefixtures("xfail_fairness_tests")
def test_modelfairness_performance(init_lens_multiclass, metrics, evaluator):
    lens, temp_file, gov = init_lens_multiclass
    eval = evaluator(metrics)
    lens.add(eval)
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))


@pytest.mark.xfail(reason="multiclass format is not supported")
def test_threshold_performance(init_lens_multiclass):
    lens, temp_file, gov = init_lens_multiclass
    lens.add(Performance(["roc_curve", "precision_recall_curve"]))
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))


@pytest.mark.parametrize(
    "evaluator",
    [
        DataProfiler,
        ModelEquity,
        Security,
        Deepchecks,
        ModelProfiler,
    ],
    ids=[
        "DataProfiler",
        "ModelEquity",
        "Security",
        "Deepchecks",
        "ModelProfiler",
    ],
)
def test_generic_evaluator(init_lens_multiclass, evaluator):
    """
    Any evaluator not requiring specific treatment can be tested here
    """
    lens, temp_file, gov = init_lens_multiclass
    lens.add(evaluator())
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))
