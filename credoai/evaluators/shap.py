from typing import Dict, List, Optional
from credoai.evaluators import Evaluator
from credoai.evidence import TableContainer
from credoai.evaluators.utils.validation import check_requirements_existence
from shap import Explainer, Explanation
from pandas import DataFrame, concat
from numpy import mean, abs


class ShapValues(Evaluator):
    def __init__(self, samples_ind: Optional[List[int]] = None):
        super().__init__()
        self.samples_ind = samples_ind

    name = "Shap"
    required_artifacts = ["assessment_data", "model"]

    def _setup(self):
        self.X = self.assessment_data.X
        self.model = self.model
        return self

    def _validate_arguments(self):
        check_requirements_existence(self)

    def evaluate(self):
        ## Overall stats
        explainer = Explainer(self.model.predict, self.X)
        self.shap_values = explainer(self.X)
        res = self._get_overall_shap_contributions()
        res.name = "Summary of Shap statistics"
        self.results = [TableContainer(res, **self.get_container_info())]

        ## Sample specific results
        labels = {"ordered_feature_names": self.shap_values.feature_names}
        if self.samples_ind:
            ind_res = self._get_mult_sample_shapley_values()
            self.results += [
                TableContainer(ind_res, **self.get_container_info(labels=labels))
            ]
        return self

    def _get_overall_shap_contributions(self) -> DataFrame:
        """
        Calculate overall SHAP contributions for a dataset.

        The output of SHAP package provides Shapley values for each sample in a
        dataset. To summarise the contribution of each feature in a dataset, the
        samples contributions need to be aggregated.

        For each of the features, this method provides: mean, max, minimum and the
        mean of the absolute value of the samples shapley values.

        Returns
        -------
        DataFrame
            Summary of the shapley values across the full dataset.
        """
        values_df = DataFrame(self.shap_values.values)
        values_df.columns = self.shap_values.feature_names
        stats_1 = values_df.apply(["mean", "min", "max"]).T
        stats_2 = values_df.apply(lambda x: mean(abs(x)))
        stats_2.name = "mean(|x|)"
        final = concat([stats_1, stats_2], axis=1)
        final = final.sort_values("mean(|x|)", ascending=False)
        return final

    def _get_mult_sample_shapley_values(self) -> DataFrame:
        """
        Return shapley values for multiple samples from the dataset.

        Returns
        -------
        DataFrame
            Columns:
                values -> shap values
                ref_value -> Reference value for the shap values
                    (generally the same across the dataset)
                sample_pos -> Position of the sample in the dataset
        """
        all_sample_shaps = []
        for ind in self.samples_ind:
            all_sample_shaps.append(
                {
                    **self._get_single_sample_values(self.shap_values[ind]),
                    **{"sample_pos": ind},
                }
            )
        res = DataFrame(all_sample_shaps)
        res.name = "Shap values for specific samples"
        return res

    @staticmethod
    def _get_single_sample_values(sample_shap: Explanation) -> Dict:
        """
        Returns shapley values for a specific sample in the dataset

        Parameters
        ----------
        shap_values : Explanation
            Explainer object output for a specific sample.
        sample_ind : int
            Position (row number) of the sample of interest in the dataset
            provided to the Explainer.

        Returns
        -------
        dict
            keys: values, ref_value
            Containes shapley values for the sample, and the reference value.
            The model prediction for the sample is equal to: ref_value + sum(values)
        """

        return {
            "values": sample_shap.values,
            "ref_value": sample_shap.base_values,
        }
