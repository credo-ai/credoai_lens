from pandas import DataFrame

from credoai.artifacts import ClassificationModel
from credoai.evaluators import Evaluator
from credoai.evaluators.utils.validation import check_requirements_existence
from credoai.evidence import MetricContainer
from credoai.evidence.containers import TableContainer
from credoai.modules.credoai_metrics import population_stability_index


class FeatureDrift(Evaluator):
    """
    Measuring drift
    """

    def __init__(self, buckets=10, buckettype="bins", csi_calculation=False):
        self.bucket_number = buckets
        self.buckettype = buckettype
        self.csi_calculation = csi_calculation
        self.percentage = False
        super().__init__()

    name = "Feature Drift"

    required_artifacts = {"model", "assessment_data", "training_data"}

    def _validate_arguments(self):
        check_requirements_existence(self)

    @staticmethod
    def _create_bin_percentage(train, assess):
        len_training = len(train)
        len_assessment = len(assess)
        train_bin_perc = train.value_counts() / len_training
        assess_bin_perc = assess.value_counts() / len_assessment
        return train_bin_perc, assess_bin_perc

    def _setup(self):
        # Default prediction to predict method
        prediction_method = self.model.predict
        if isinstance(self.model, ClassificationModel):
            if hasattr(self.model, "predict_proba"):
                prediction_method = self.model.predict_proba
            else:
                self.percentage = True

        self.expected_prediction = prediction_method(self.training_data.X)
        self.actual_prediction = prediction_method(self.assessment_data.X)

        # Create the bins manually for categorical prediction if predict_proba
        # is not available.
        if self.percentage:
            (
                self.expected_prediction,
                self.actual_prediction,
            ) = self._create_bin_percentage(
                self.expected_prediction, self.actual_prediction
            )

    def evaluate(self):
        prediction_psi = self._calculate_psi_on_prediction()
        self.results = [MetricContainer(prediction_psi, self.get_container_info())]
        if self.csi_calculation:
            csi = self._calculate_csi()
            self.results.append(TableContainer(csi, self.get_container_info()))
        return self

    def _calculate_psi_on_prediction(self):
        psi = population_stability_index(
            self.expected_prediction,
            self.actual_prediction,
            percentage=self.percentage,
            buckets=self.bucket_number,
            buckettype=self.buckettype,
        )
        res = DataFrame({"value": psi, "type": "population_stability_index"}, index=[0])
        return res

    def _calculate_csi(self):
        columns_names = list(self.assessment_data.X.columns)
        psis = {}
        for col_name in columns_names:
            train_data = self.training_data.X[col_name]
            assess_data = self.assessment_data.X[col_name]
            if self.assessment_data.X[col_name].dtype == "category":
                train, asess = self._create_bin_percentage(train_data, assess_data)
                psis[col_name] = population_stability_index(train, asess, True)
            else:
                psis[col_name] = population_stability_index(train_data, assess_data)
        psis = DataFrame.from_dict(psis, orient="index")
        psis.columns = ["value"]
        psis.name = "Characteristic Stability Index"
        return psis
