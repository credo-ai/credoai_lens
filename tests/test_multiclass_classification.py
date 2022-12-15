"""
Testing protocols for the Lens package. Tested functionalities:

    1. Individual evaluator runs within Lens framework
    2. Full run of Lens pipeline with multiple evaluators
"""


import pytest

from credoai.evaluators import (
    ModelFairness,
    Performance,
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
