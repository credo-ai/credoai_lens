"""
Testing protocols for the Lens package. Tested functionalities:
    1. General inits
    2. Individual evaluator runs
"""

from abc import ABC

import pytest
from pandas import DataFrame

from credoai.artifacts import ComparisonData, ComparisonModel, TabularData
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

from connect.governance import Governance

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


@pytest.fixture(scope="function")
def temp_file(tmp_path):
    d = tmp_path / "test.json"
    d.touch()
    return d


@pytest.fixture(scope="function")
def init_lens_classification(
    classification_model,
    classification_assessment_data,
    classification_train_data,
    temp_file,
):
    gov = Governance()
    my_pipeline = Lens(
        model=classification_model,
        assessment_data=classification_assessment_data,
        training_data=classification_train_data,
        governance=gov,
    )
    return my_pipeline, temp_file, gov


@pytest.fixture(scope="function")
def init_lens_credit(
    credit_classification_model,
    credit_assessment_data,
    credit_training_data,
    temp_file,
):
    gov = Governance()
    my_pipeline = Lens(
        model=credit_classification_model,
        assessment_data=credit_assessment_data,
        training_data=credit_training_data,
        governance=gov,
    )
    return my_pipeline, temp_file, gov


@pytest.fixture(scope="function")
def init_lens_fairness(temp_file):
    df = DataFrame(
        {
            "rankings": [1, 2, 3, 4, 5, 6, 7, 8],
            "scores": [10, 8, 7, 6, 2, 2, 1, 1],
            "sensitive_features": ["f", "f", "m", "m", "f", "m", "f", "f"],
        }
    )
    data = TabularData(
        name="ranks",
        y=df[["rankings", "scores"]],
        sensitive_features=df[["sensitive_features"]],
    )
    expected_results = DataFrame(
        {
            "value": [0.11, 0.20, 0.90, 0.67, 0.98, 0.65, 0.59],
            "type": [
                "skew_parity_difference",
                "ndkl",
                "demographic_parity_ratio",
                "balance_ratio",
                "score_parity_ratio",
                "score_balance_ratio",
                "relevance_parity_ratio",
            ],
            "subtype": ["score"] * 7,
        }
    )
    gov = Governance()
    pipeline = Lens(assessment_data=data, governance=gov)

    return pipeline, temp_file, gov, expected_results


@pytest.fixture(scope="function")
def init_lens_identityverification(temp_file):
    source_subject_id = 4 * ["s0"] + 5 * ["s1"] + ["s2"]
    source_subject_data_sample = 4 * ["s00"] + 3 * ["s10"] + 2 * ["s11"] + ["s20"]
    target_subject_id = ["s1"] + ["s1", "s2", "s3"] * 2 + ["s2", "s3", "s3"]
    target_subject_data_sample = (
        ["s10"] + ["s11", "s20", "s30"] * 2 + ["s20"] + ["s30"] * 2
    )
    pairs = DataFrame(
        {
            "source-subject-id": source_subject_id,
            "source-subject-data-sample": source_subject_data_sample,
            "target-subject-id": target_subject_id,
            "target-subject-data-sample": target_subject_data_sample,
        }
    )

    subjects_sensitive_features = DataFrame(
        {
            "subject-id": ["s0", "s1", "s2", "s3"],
            "gender": ["female", "male", "female", "female"],
        }
    )

    expected_results_perf = DataFrame(
        {
            "value": [0.33, 1.00],
            "type": ["false_match_rate", "false_non_match_rate"],
            "subtype": ["score"] * 2,
        }
    )

    expected_results_fair = DataFrame(
        {
            "gender": ["female", "male", "female", "male"],
            "type": [
                "false_match_rate",
                "false_match_rate",
                "false_non_match_rate",
                "false_non_match_rate",
            ],
            "value": [0, 0, 0, 1],
        }
    )
    expected_results = {"fair": expected_results_fair, "perf": expected_results_perf}
    similarity_scores = [31.5, 16.7, 20.8, 84.4, 12.0, 15.2, 45.8, 23.5, 28.5, 44.5]

    credo_data = ComparisonData(
        name="face-data",
        pairs=pairs,
        subjects_sensitive_features=subjects_sensitive_features,
    )
    credo_model = DummyComparisonModel(
        name="face-compare", compare_output=similarity_scores
    )
    gov = Governance()
    pipeline = Lens(model=credo_model, assessment_data=credo_data, governance=gov)

    return pipeline, temp_file, gov, expected_results


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


def test_privacy(init_lens_credit):
    lens, temp_file, gov = init_lens_credit
    lens.add(Privacy(attack_feature="MARRIAGE"))
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))


def test_data_fairness(init_lens_classification):
    lens, temp_file, gov = init_lens_classification
    lens.add(DataFairness())
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))


def test_data_profiler(init_lens_classification):
    lens, temp_file, gov = init_lens_classification
    lens.add(DataProfiler())
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))


def test_model_equity(init_lens_classification):
    lens, temp_file, gov = init_lens_classification
    lens.add(ModelEquity())
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))


def test_data_equity(init_lens_classification):
    lens, temp_file, gov = init_lens_classification
    lens.add(DataEquity())
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))


def test_security(init_lens_classification):
    lens, temp_file, gov = init_lens_classification
    lens.add(Security())
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


def test_deepchecks(init_lens_classification):
    lens, temp_file, gov = init_lens_classification
    lens.add(Deepchecks())
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
