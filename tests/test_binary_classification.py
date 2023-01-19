"""
Testing protocols for the Lens package.

Testing evaluators behavior when dealing with binary classification
data/models.
"""

import pytest

from credoai.evaluators import (
    DataEquity,
    DataFairness,
    DataProfiler,
    Deepchecks,
    FeatureDrift,
    IdentityVerification,
    ModelEquity,
    ModelFairness,
    ModelProfiler,
    Performance,
    Privacy,
    Security,
    ShapExplainer,
)
from credoai.evaluators.ranking_fairness import RankingFairness
from credoai.lens import Lens

from credoai.utils import ValidationError

##################################################
#################### Init ########################
##################################################

TEST_METRICS = [
    ["false_negative_rate"],
    ["average_precision_score"],
    ["false_negative_rate", "average_precision_score"],
    ["precision_score", "equal_opportunity"],
    ["roc_curve", "gain_chart"],
    ["false_negative_rate", "average_precision_score", "equal_opportunity"],
]
TEST_METRICS_IDS = [
    "binary_metric",
    "probability_metric",
    "binary_and_probability",
    "fairness",
    "threshold",
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
    [
        DataFairness,
        DataProfiler,
        ModelEquity,
        DataEquity,
        Security,
        Deepchecks,
        ModelProfiler,
    ],
    ids=[
        "DataFairness",
        "DataProfiler",
        "ModelEquity",
        "DataEquity",
        "Security",
        "Deepchecks",
        "ModelProfiler",
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


@pytest.mark.parametrize(
    "samples_ind,background_samples,background_kmeans",
    [([], 5, None), ([], None, 5), ([1, 2, 7], None, 5)],
    ids=["Samples", "KMeans", "Ind_Samples"],
)
def test_shap(
    init_lens_classification, samples_ind, background_samples, background_kmeans
):
    lens, temp_file, gov = init_lens_classification
    eval = ShapExplainer(
        samples_ind=samples_ind,
        background_samples=background_samples,
        background_kmeans=background_kmeans,
    )
    lens.add(eval)
    lens.run()
    pytest.assume(lens.get_results())
    pytest.assume(lens.get_evidence())
    pytest.assume(lens.send_to_governance())
    pytest.assume(not gov._file_export(temp_file))


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


def test_fairness_validation_no_sens_feat(init_lens_classification):
    """
    Tests to ensure Lens will not allow running evaluators that require sensitive features without
    any sensitive features specified
    """
    lens, _, _ = init_lens_classification
    lens.assessment_data.sensitive_features = None
    lens.sens_feat_names = []
    lens.training_data = None
    with pytest.raises(Exception) as e_info:
        lens.add(ModelFairness(["accuracy_score"]))

    pytest.assume(type(e_info.value) == ValidationError)


@pytest.mark.parametrize(
    "evaluator",
    [
        DataFairness,
        ModelEquity,
        DataEquity,
    ],
    ids=[
        "DataFairness",
        "ModelEquity",
        "DataEquity",
    ],
)
def test_generic_validation_no_sens_feat(init_lens_classification, evaluator):
    """
    Tests to ensure Lens will not allow running evaluators that require sensitive features without
    any sensitive features specified
    """
    lens, _, _ = init_lens_classification
    lens.assessment_data.sensitive_features = None
    lens.sens_feat_names = []
    lens.training_data = None
    with pytest.raises(Exception) as e_info:
        lens.add(evaluator())

    pytest.assume(type(e_info.value) == ValidationError)


def test_ranking_validation_no_sens_feat(init_lens_fairness):
    lens, _, _, _ = init_lens_fairness
    lens.assessment_data.sensitive_features = None
    lens.sens_feat_names = []
    lens.training_data = None
    with pytest.raises(Exception) as e_info:
        lens.add(RankingFairness(k=5))
    pytest.assume(type(e_info.value) == ValidationError)


def test_print_results(init_lens_classification):
    lens, _, _ = init_lens_classification
    lens.add(Performance(["accuracy_score"]))
    lens.run()
    # pytest.assume(lens.get_results())
    pytest.assume(not lens.print_results())
