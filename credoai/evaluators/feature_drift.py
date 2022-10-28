"""Feature Drift evaluator"""
from credoai.artifacts import ClassificationModel
from credoai.evaluators import Evaluator
from credoai.evaluators.utils.validation import check_requirements_existence
from credoai.evidence import MetricContainer
from credoai.evidence.containers import TableContainer
from credoai.modules.credoai_metrics import population_stability_index
from pandas import DataFrame, Series


class FeatureDrift(Evaluator):
    """
    Measure Feature Drift using population stability index.

    This evaluator measures feature drift in:

    1. Model prediction: the prediction for the assessment dataset is compared
        to the prediction for the training dataset.
        In the case of classifiers, the prediction is performed with predict proba if available.
        If it is not available, the prediction is treated like a categorical variable, see the
        processing of categorical variables in the item below.

    2. Dataset features: 1 to 1 comparison across all features for the datasets. This is also
    referred to as "characteristic stability index" (CSI).
        - Numerical features are directly fed into the population_stability_index metric, and
        binned according to the parameters specified at init time.
        - Categorical features percentage distribution is manually calculated. The % amount of
        samples per each class is calculated and then fed into the population_stability_index metric.
        The percentage flag in the metric is set to True, to bypass the internal binning process.


    Parameters
    ----------
    buckets : int, optional
        Number of buckets to consider to bin the predictions, by default 10
    buckettype :  Literal["bins", "quantiles"]
        Type of strategy for creating buckets, bins splits into even splits,
        quantiles splits into quantiles buckets, by default "bins"
    csi_calculation : bool, optional
        Calculate characteristic stability index, i.e., PSI for all features in the datasets,
        by default False
    """

    def __init__(self, buckets: int = 10, buckettype="bins", csi_calculation=False):

        self.bucket_number = buckets
        self.buckettype = buckettype
        self.csi_calculation = csi_calculation
        self.percentage = False
        super().__init__()

    required_artifacts = {"model", "assessment_data", "training_data"}

    def _validate_arguments(self):
        check_requirements_existence(self)

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
        self.results = [MetricContainer(prediction_psi, **self.get_container_info())]
        if self.csi_calculation:
            csi = self._calculate_csi()
            self.results.append(TableContainer(csi, **self.get_container_info()))
        return self

    def _calculate_psi_on_prediction(self) -> DataFrame:
        """
        Calculate the psi index on the model prediction.

        Returns
        -------
        DataFrame
            Formatted for metric container.
        """
        psi = population_stability_index(
            self.expected_prediction,
            self.actual_prediction,
            percentage=self.percentage,
            buckets=self.bucket_number,
            buckettype=self.buckettype,
        )
        res = DataFrame({"value": psi, "type": "population_stability_index"}, index=[0])
        return res

    def _calculate_csi(self) -> DataFrame:
        """
        Calculate psi for all the columns in the dataframes.

        Returns
        -------
        DataFrame
            Formatted for the table container.
        """
        columns_names = list(self.assessment_data.X.columns)
        psis = {}
        for col_name in columns_names:
            train_data = self.training_data.X[col_name]
            assess_data = self.assessment_data.X[col_name]
            if self.assessment_data.X[col_name].dtype == "category":
                train, assess = self._create_bin_percentage(train_data, assess_data)
                psis[col_name] = population_stability_index(train, assess, True)
            else:
                psis[col_name] = population_stability_index(train_data, assess_data)
        psis = DataFrame.from_dict(psis, orient="index")
        psis = psis.reset_index()
        psis.columns = ["feature_names", "value"]
        psis.name = "Characteristic Stability Index"
        return psis

    @staticmethod
    def _create_bin_percentage(train: Series, assess: Series) -> tuple:
        """
        In case of categorical values proceed to count the instances
        of each class and divide by the total amount of samples to get
        the ratios.

        Parameters
        ----------
        train : Series
            Array of values, dtype == category
        assess : Series
            Array of values, dtype == category

        Returns
        -------
        tuple
            Class percentages for both arrays
        """
        len_training = len(train)
        len_assessment = len(assess)
        train_bin_perc = train.value_counts() / len_training
        assess_bin_perc = assess.value_counts() / len_assessment
        return train_bin_perc, assess_bin_perc
