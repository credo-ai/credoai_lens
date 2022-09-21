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
    """
    Base evaluator class

    This takes in the initialized lens fixture and defines standardized tests
    for each evaluator.
    """

    @abstractmethod
    def test_add(self):
        """
        Tests that the step was effectively added to the pipeline.

        Depending on evaluator requirements (data/sensitive feature) the
        assert statement on the length of the pipeline changes.
        """
        ...

    @abstractmethod
    def test_run(self):
        """
        Tests that the pipeline run, checking for results presence.
        """
        ...


class TestModelFairness(Base_Evaluator_Test):
    evaluator = ModelFairness(metrics=["precision_score"])

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 4

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()


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


class TestEquity(Base_Evaluator_Test):
    evaluator = Equity()

    def test_add(self):
        self.pipeline.add(self.evaluator)
        assert len(self.pipeline.pipeline) == 2

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
        assert len(self.pipeline.pipeline) == 2

    def test_run(self):
        self.pipeline.run()
        assert self.pipeline.get_results()


def test_bulk_pipeline_run(credo_model, assessment_data, train_data):
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
        model=credo_model,
        assessment_data=assessment_data,
        training_data=train_data,
        pipeline=pipe_structure,
    )
    my_pipeline.run()
    assert my_pipeline.get_results()


def test_automatic_run(credo_model, assessment_data, train_data):
    my_pipeline = Lens(
        model=credo_model,
        assessment_data=assessment_data,
        training_data=train_data,
    )
    my_pipeline.run()
    assert my_pipeline.get_results()
