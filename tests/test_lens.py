"""
Testing protocols for the Lens package. Tested functionalities:
    1. General inits
    2. Individual evaluator runs
"""

from abc import ABC

import pytest
from credoai.artifacts import TabularData, ComparisonData, ComparisonModel
from credoai.evaluators import (
    DataEquity,
    DataFairness,
    DataProfiler,
    FeatureDrift,
    ModelEquity,
    ModelFairness,
    Performance,
    Privacy,
    Security,
    FeatureDrift,
    Deepchecks,
    IdentityVerification
)
from credoai.evaluators.ranking_fairness import RankingFairness
from credoai.lens import Lens
from pandas import DataFrame

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


@pytest.fixture(scope="class")
def init_lens(
    classification_model,
    classification_assessment_data,
    classification_train_data,
    request,
):
    my_pipeline = Lens(
        model=classification_model,
        assessment_data=classification_assessment_data,
        training_data=classification_train_data,
    )
    request.cls.pipeline = my_pipeline


@pytest.mark.usefixtures("init_lens")
class Base_Evaluator_Test(ABC):
    """
    Base evaluator class

    This takes in the initialized lens fixture and defines standardized tests
    for each evaluator.
    """

    ...


@pytest.mark.parametrize("metrics", TEST_METRICS, ids=TEST_METRICS_IDS)
def test_model_fairness(
    classification_model,
    classification_assessment_data,
    classification_train_data,
    metrics,
):
    lens = Lens(
        model=classification_model,
        assessment_data=classification_assessment_data,
        training_data=classification_train_data,
    )
    evaluator = ModelFairness(metrics)
    lens.add(evaluator)
    lens.run()
    assert lens.get_results()


def test_privacy(
    credit_classification_model, credit_assessment_data, credit_training_data
):
    lens = Lens(
        model=credit_classification_model,
        assessment_data=credit_assessment_data,
        training_data=credit_training_data,
    )
    lens.add(Privacy(attack_feature="MARRIAGE"))
    lens.run()
    assert lens.get_results()


class TestDataFairness(Base_Evaluator_Test):
    evaluator = DataFairness()

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 4

    def test_run(self):
        self.pipeline.run()
        self.pipeline.get_results()
        assert True


class TestDataProfiler(Base_Evaluator_Test):
    evaluator = DataProfiler()

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 2

    def test_run(self):
        self.pipeline.run()
        self.pipeline.get_results()
        assert True


class TestModelEquity(Base_Evaluator_Test):
    evaluator = ModelEquity()

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 2

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()


class TestDataEquity(Base_Evaluator_Test):
    evaluator = DataEquity()

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 4

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()


class TestSecurity(Base_Evaluator_Test):
    evaluator = Security()

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 1

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()


@pytest.mark.parametrize("metrics", TEST_METRICS, ids=TEST_METRICS_IDS)
def test_performance(
    credit_classification_model, credit_assessment_data, credit_training_data, metrics
):
    lens = Lens(
        model=credit_classification_model,
        assessment_data=credit_assessment_data,
        training_data=credit_training_data,
    )
    evaluator = Performance(metrics)
    lens.add(evaluator)
    lens.run()
    assert lens.get_results()


class TestThresholdPerformance(Base_Evaluator_Test):
    evaluator = Performance(["roc_curve"])

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 1

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()


class TestThresholdPerformanceMultiple(Base_Evaluator_Test):
    evaluator = Performance(["roc_curve", "precision_recall_curve"])

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 1

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()


class TestFeatureDrift(Base_Evaluator_Test):
    evaluator = FeatureDrift(csi_calculation=True)

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 1

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()


class TestDeepchecks(Base_Evaluator_Test):
    evaluator = Deepchecks()

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 1

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()


class TestRankingFairnes:
    evaluator = RankingFairness(k=5)

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
    pipeline = Lens(assessment_data=data)

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 1

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()

    def test_results(self):
        results = self.pipeline.get_results()[0]["results"][0].round(2)
        results = results.reset_index(drop=True)
        assert results.equals(self.expected_results)


class TestIdentityVerification:
    evaluator = IdentityVerification(similarity_thresholds=[60, 99])

    pairs = DataFrame({
        'source-subject-id': ['s0', 's0', 's0', 's0', 's1', 's1', 's1', 's1', 's1', 's2'],
        'source-subject-data-sample': ['s00', 's00', 's00', 's00', 's10', 's10', 's10', 's11', 's11', 's20'],
        'target-subject-id': ['s1', 's1', 's2', 's3', 's1', 's2', 's3', 's2', 's3', 's3'],
        'target-subject-data-sample': ['s10', 's11', 's20', 's30', 's11', 's20', 's30', 's20', 's30', 's30']
    })

    subjects_sensitive_features = DataFrame({
        'subject-id': ['s0', 's1', 's2', 's3'],
        'gender': ['female', 'male', 'female', 'female']
    })

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
            "type": ["false_match_rate", "false_match_rate", "false_non_match_rate", "false_non_match_rate"],
            "value": [0, 0, 0, 1],
        }
    )

    class FaceCompare:
        def compare(self, pairs):
            similarity_scores = [31.5, 16.7, 20.8, 84.4, 12.0, 15.2, 45.8, 23.5, 28.5, 44.5]
            return similarity_scores

    face_compare = FaceCompare()

    credo_data = ComparisonData(
        name="face-data",
        pairs=pairs,
        subjects_sensitive_features=subjects_sensitive_features
        )

    credo_model = ComparisonModel(
        name="face-compare", 
        model_like=face_compare
        )

    pipeline = Lens(model=credo_model, assessment_data=credo_data)

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 1

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()

    def test_get_results(self):
        results = self.pipeline.get_results()[0]['results']
        assert len(results) == 8

    def test_results_performance(self):
        results_perf = self.pipeline.get_results()[0]['results'][0].round(2)
        results_perf = results_perf.reset_index(drop=True)
        assert results_perf.equals(self.expected_results_perf)

    def test_results_fairness(self):
        results_fair = self.pipeline.get_results()[0]['results'][-1]
        results_fair['value'] = results_fair['value'].astype(int)
        results_fair = results_fair.reset_index(drop=True)
        assert results_fair.equals(self.expected_results_fair)


def test_bulk_pipeline_run(
    classification_model, classification_assessment_data, classification_train_data
):
    """
    Testing the passing of the list of evaluator works
    and the pipeline is running.
    """
    pipe_structure = [
        Security(),
        DataProfiler(),
        DataFairness(),
    ]
    my_pipeline = Lens(
        model=classification_model,
        assessment_data=classification_assessment_data,
        training_data=classification_train_data,
        pipeline=pipe_structure,
    )
    my_pipeline.run()
    assert my_pipeline.get_results()


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
