"""
Testing protocols for the Lens package. Tested functionalities:
    1. General inits
    2. Individual evaluator runs
"""

from credoai.lens import Lens
import pytest
from credoai.evaluators import ModelFairness, Privacy


@pytest.fixture(scope="class")
def init_lens(credo_model, assessment_data, train_data, request):
    my_pipeline = Lens(
        model=credo_model, assessment_data=assessment_data, training_data=train_data
    )
    request.cls.pipeline = my_pipeline


@pytest.mark.usefixtures("init_lens")
class Base_Evaluator_Test:
    pass


class TestModelFairness(Base_Evaluator_Test):
    def test_add(self):
        self.pipeline.add(ModelFairness(metrics=["false positive rate"]))
        assert len(self.pipeline.pipeline) == 2

    def test_run(self):
        self.pipeline.run()
        assert True


class TestPrivacy(Base_Evaluator_Test):
    def test_add(self):
        self.pipeline.add(Privacy(attack_feature="MARRIAGE"))
        assert len(self.pipeline.pipeline) == 1

    def test_run(self):
        self.pipeline.run()
        assert True
