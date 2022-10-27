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
    DataProfiler,
    ModelEquity,
    ModelFairness,
    Performance,
)
from credoai.lens import Lens
from pandas import DataFrame

TEST_METRICS = [["r2_score"]]
TEST_METRICS_IDS = ["regression_metric"]


@pytest.fixture(scope="class")
def init_lens(
    regression_model,
    regression_assessment_data,
    regression_train_data,
    request,
):
    my_pipeline = Lens(
        model=regression_model,
        assessment_data=regression_assessment_data,
        training_data=regression_train_data,
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


class TestPerformance(Base_Evaluator_Test):
    @pytest.mark.parametrize("metrics", TEST_METRICS, ids=TEST_METRICS_IDS)
    def test_full_run(self, metrics):
        evaluator = Performance(metrics)
        self.pipeline.add(evaluator)
        self.pipeline.run()
        assert len(self.pipeline.pipeline) == 1
        assert self.pipeline.get_results()
        self.pipeline.pipeline = {}
