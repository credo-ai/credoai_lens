"""
Testing protocols for the Lens package. Tested functionalities:
    1. General inits
    2. Individual evaluator runs
"""


import pytest

from credoai.artifacts.model.comparison_model import DummyComparisonModel
from credoai.evaluators import (
    DataEquity,
    DataFairness,
    DataProfiler,
    Deepchecks,
    FeatureDrift,
    IdentityVerification,
    ModelEquity,
    ModelFairness,
    Performance,
    Privacy,
    Security,
)
from credoai.evaluators.ranking_fairness import RankingFairness
from credoai.lens import Lens


##################################################
#################### Init ########################
##################################################

TEST_METRICS = [
    ["false_negative_rate"],
    ["average_precision_score"],
    ["false_negative_rate", "average_precision_score"],
    ["precision_score", "equal_opportunity"],
    ["false_negative_rate", "average_precision_score", "equal_opportunity"],
]
TEST_METRICS_IDS = [
    "binary_metric",
    "probability_metric",
    "binary_and_probability",
    "fairness",
    "all_types",
]


##################################################
#################### Tests #######################
##################################################


@pytest.mark.parametrize(
    "evaluator", [ModelFairness, Performance], ids=["Model Fairness", "Performance"]
)
@pytest.mark.parametrize("metrics", TEST_METRICS, ids=TEST_METRICS_IDS)
def test_modelfairness_performance(init_lens_classification, metrics, evaluator):
    lens, temp_file, gov = init_lens_classification
    eval = evaluator(metrics)
    lens.add(eval)
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))


def test_threshold_performance(init_lens_classification):
    lens, temp_file, gov = init_lens_classification
    lens.add(Performance(["roc_curve", "precision_recall_curve"]))
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))


@pytest.mark.parametrize(
    "evaluator",
    [DataFairness, DataProfiler, ModelEquity, DataEquity, Security, Deepchecks],
    ids=[
        "DataFairness",
        "DataProfiler",
        "ModelEquity",
        "DataEquity",
        "Security",
        "Deepchecks",
    ],
)
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


def test_privacy(init_lens_credit):
    lens, temp_file, gov = init_lens_credit
    lens.add(Privacy(attack_feature="MARRIAGE"))
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))


def test_feature_drift(init_lens_classification):
    lens, temp_file, gov = init_lens_classification
    lens.add(FeatureDrift(csi_calculation=True))
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))


def test_ranking_fairness(init_lens_fairness):
    lens, temp_file, gov, expected_results = init_lens_fairness
    lens.add(RankingFairness(k=5))
    lens.run()
    results = lens.get_results()[0]["results"][0].round(2)
    results = results.reset_index(drop=True)
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))
    pytest.assume(results.equals(expected_results))


def test_identity_verification(init_lens_identityverification):
    lens, temp_file, gov, expected_results = init_lens_identityverification
    lens.add(IdentityVerification(similarity_thresholds=[60, 99]))
    lens.run()
    # Get peformance results
    results_perf = lens.get_results()[0]["results"][0].round(2)
    results_perf = results_perf.reset_index(drop=True)
    # Get fairness results
    results_fair = lens.get_results()[0]["results"][-4]
    results_fair["value"] = results_fair["value"].astype(int)
    results_fair = results_fair.reset_index(drop=True)
    # Assertions
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))
    pytest.assume(results_perf.equals(expected_results["perf"]))
    pytest.assume(results_fair.equals(expected_results["fair"]))


def test_bulk_pipeline_run(init_lens_classification):
    """
    Testing the passing of the list of evaluator works
    and the pipeline is running.
    """
    lens, temp_file, gov = init_lens_classification
    lens.add(Security())
    lens.add(DataProfiler())
    lens.add(DataFairness())
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))


@pytest.mark.xfail(raises=RuntimeError)
def test_empty_pipeline_run(
    classification_model, classification_assessment_data, classification_train_data
):
    my_pipeline = Lens(
        model=classification_model,
        assessment_data=classification_assessment_data,
        training_data=classification_train_data,
    )
    my_pipeline.run()
