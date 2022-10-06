"""
Testing protocols for the Lens package. Tested functionalities:
    1. General inits
    2. Individual evaluator runs
"""

from abc import ABC, abstractmethod

import pytest
from credoai.artifacts import TabularData
from credoai.evaluators import (
    DataEquity,
    DataFairness,
    DataProfiling,
    ModelEquity,
    ModelFairness,
    Performance,
    Privacy,
    Security,
)
from credoai.evaluators.ranking_fairness import RankingFairness
from credoai.lens import Lens
from pandas import DataFrame

TEST_METRICS = [
    ["false_negative_rate"],
    ["average_precision_score"],
    ["false_negative_rate", "average_precision_score"],
]
TEST_METRICS_IDS = ["binary_metric", "probability_metric", "both"]


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


class TestModelFairness(Base_Evaluator_Test):
    @pytest.mark.parametrize("metrics", TEST_METRICS, ids=TEST_METRICS_IDS)
    def test_full_run(self, metrics):
        evaluator = ModelFairness(metrics)
        self.pipeline.add(evaluator)
        self.pipeline.run()
        assert len(self.pipeline.pipeline) == 4
        assert self.pipeline.get_results()
        self.pipeline.pipeline = {}


class TestPrivacy(Base_Evaluator_Test):
    evaluator = Privacy(attack_feature="experience")

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 1

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()


class TestDataFairness(Base_Evaluator_Test):
    evaluator = DataFairness()

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 4

    def test_run(self):
        self.pipeline.get_results()
        assert True


class TestDataProfiling(Base_Evaluator_Test):
    evaluator = DataProfiling()

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 2

    def test_run(self):
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


class TestPerformance(Base_Evaluator_Test):
    @pytest.mark.parametrize("metrics", TEST_METRICS, ids=TEST_METRICS_IDS)
    def test_full_run(self, metrics):
        evaluator = Performance(metrics)
        self.pipeline.add(evaluator)
        self.pipeline.run()
        assert len(self.pipeline.pipeline) == 1
        assert self.pipeline.get_results()
        self.pipeline.pipeline = {}


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


class TestRankingFairnes:
    evaluator = RankingFairness()

    df = DataFrame(
        {
            "rankings": [1, 2, 3, 4, 5, 6, 7],
            "sensitive_features": ["f", "f", "m", "m", "f", "m", "f"],
        }
    )
    data = TabularData(
        name="ranks", y=df[["rankings"]], sensitive_features=df[["sensitive_features"]]
    )
    expected_results = DataFrame(
        {
            "value": [0.86, 1.14, 0.32],
            "type": ["minimum_skew", "maximum_skew", "NDKL"],
            "subtype": ["score"] * 3,
        }
    )
    pipeline = Lens(assessment_data=data)

    def test_add(self):
        self.pipeline.add(self.evaluator, "dummy")
        assert len(self.pipeline.pipeline) == 1

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()

    def test_results(self):
        results = self.pipeline.get_results()["dummy"][0].round(2)
        results = results.reset_index(drop=True)
        assert results.equals(self.expected_results)


def test_bulk_pipeline_run(
    classification_model, classification_assessment_data, classification_train_data
):
    """
    Testing the passing of the list of evaluator works
    and the pipeline is running.
    """
    pipe_structure = [
        (Security(), "Security assessment"),
        (DataProfiling(), "Profiling test data"),
        (DataFairness(), "Test data Fairness"),
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
