from credoai.evaluators import Evaluator
from credoai.evidence import TableContainer
from shap import Explainer, Explanation
from pandas import DataFrame, concat
from numpy import mean, abs


class ShapValues(Evaluator):
    def __init__(self):
        super().__init__()

    name = "Shap"
    required_artifacts = ["assessment_data", "model"]

    def _setup(self):
        self.X = self.assessment_data.X
        self.model = self.model
        return self

    def _validate_arguments(self):
        pass

    def evaluate(self):
        explainer = Explainer(self.model.predict, self.X)
        shap_values = explainer(self.X)
        res = self._get_overall_shap_contributions(shap_values=shap_values)
        res.name = "Summary of Shap statistics"
        self.results = [TableContainer(res, **self.get_container_info())]
        return self

    @staticmethod
    def _get_overall_shap_contributions(shap_values: Explanation) -> DataFrame:
        """
        Calculate overall SHAP contributions for a dataset.

        The output of SHAP package provides Shapley values for each sample in a
        dataset. To summarise the contribution of each feature in a dataset, the
        samples contributions need to be aggregated.

        For each of the features, this method provides: mean, max, minimum and the
        mean of the absolute value of the samples shapley values.


        Parameters
        ----------
        shap_values : Explanation
            The result of calling the SHAP Explainer on a specific dataset. This object
            containes all the information about shapley values for all the samples.

        Returns
        -------
        DataFrame
            Summary of the shapley values across the full dataset.
        """
        values_df = DataFrame(shap_values.values)
        values_df.columns = shap_values.feature_names
        stats_1 = values_df.apply(["mean", "min", "max"]).T
        stats_2 = values_df.apply(lambda x: mean(abs(x)))
        stats_2.name = "mean(|x|)"
        final = concat([stats_1, stats_2], axis=1)
        final = final.sort_values("mean(|x|)", ascending=False)
        return final
