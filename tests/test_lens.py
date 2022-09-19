"""
Testing protocols for the Lens package. Tested functionalities:
    1. General inits
    2. Individual evaluator runs
"""

from abc import ABC, abstractmethod
from credoai.lens import Lens
import pytest
from credoai.evaluators import DataFairness, DataProfiling, Security, Privacy
from credoai.evaluators import Performance, Equity, ModelFairness


@pytest.fixture(scope="class")
def init_lens(credo_model, assessment_data, train_data, request):
    my_pipeline = Lens(
        model=credo_model, assessment_data=assessment_data, training_data=train_data
    )
    request.cls.pipeline = my_pipeline


@pytest.mark.usefixtures("init_lens")
class Base_Evaluator_Test(ABC):
    @abstractmethod
    def test_add(self):
        ...

    @abstractmethod
    def test_run(self):
        ...


class TestModelFairness(Base_Evaluator_Test):
    evaluator = ModelFairness(metrics=["false positive rate"])

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 2

    def test_run(self):
        self.pipeline.get_results()
        assert True


class TestPrivacy(Base_Evaluator_Test):
    evaluator = Privacy(attack_feature="MARRIAGE")

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
        assert len(self.pipeline.pipeline) == 2

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


class TestEquity(Base_Evaluator_Test):
    evaluator = Equity()

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 1

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
    evaluator = Performance(["false negative rate"])

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 1

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()
