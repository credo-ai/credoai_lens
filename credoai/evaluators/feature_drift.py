from pandas import DataFrame

from credoai.artifacts import ClassificationModel, RegressionModel
from credoai.evaluators import Evaluator
from credoai.evidence import MetricContainer
from credoai.modules.credoai_metrics import population_stability_index


class FeatureDrift(Evaluator):
    """
    Measuring drift
    """

    def __init__(self, buckets=10, buckettype="bins"):
        self.bucket_number = buckets
        self.buckettype = buckettype
        super().__init__()

    name = "Feature Drift"

    required_artifacts = {"model", "assessment_data", "training_data"}

    def _setup(self):
        if isinstance(self.model, RegressionModel):
            prediction_method = self.model.predict
        if isinstance(self.model, ClassificationModel):
            prediction_method = self.model.predict_proba

        self.expected_prediction = prediction_method(self.training_data.X)
        self.actual_prediction = prediction_method(self.assessment_data.X)

    def evaluate(self):
        prediction_psi = self._calculate_psi_on_prediction()
        self.results = MetricContainer(prediction_psi, self.get_container_info())
        return self

    def _calculate_psi_on_prediction(self):
        psi = population_stability_index(
            self.expected_prediction,
            self.actual_prediction,
            buckets=self.bucket_number,
            buckettype=self.buckettype,
        )
        res = DataFrame({"value": psi, "type": "population_stability_index"})
        return res
