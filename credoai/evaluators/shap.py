from typing import Dict, List, Optional, Union

from numpy import abs, mean
from pandas import DataFrame, concat
from shap import Explainer, Explanation, kmeans

from credoai.evaluators import Evaluator
from credoai.evaluators.utils.validation import check_requirements_existence
from credoai.evidence import TableContainer
from credoai.utils.common import ValidationError


class ShapExplainer(Evaluator):
    """
    This evaluator perform the calculation of shapley values for a dataset/model,
    leveraging the SHAP package.

    It supports 2 types of assessments:
    1. Overall statistics of the shap values across all samples: mean and mean(|x|)
    2. Individual shapley values for a list of samples

    Sampling
    --------
    In order to speed up computation time, at the stage in which the SHAP explainer is
    initialized, a down sampled version of the dataset is passed to the `Explainer`
    object as background data. This is only affecting the calculation of the reference
    value, the calculation of the shap values is still performed on the full dataset.

    Two strategies for down sampling are provided:
        1. Random sampling (the default strategy): the amount of samples can be specified
            by the user.
        2. Kmeans: summarizes a dataset with k mean centroids, weighted by the number of
            data points they each represent. The amount of centroids can also be specified
            by the user.

    There is no consensus on the optimal down sampling approach. For reference, see this
    conversation: https://github.com/slundberg/shap/issues/1018


    Categorical variables
    ---------------------
    The interpretation of the results for categorical variables can be more challenging, and
    dependent on the type of encoding utilized. Ordinal or one/hot encoding can be hard to
    interpret.

    There is no agreement as to what is the best strategy as far as categorical variables are
    concerned. A good discussion on this can be found here: https://github.com/slundberg/shap/issues/451

    No restriction on feature type is imposed by the evaluator, so user discretion in the
    interpretation of shap values for categorical variables is advised.


    Parameters
    ----------
    samples_ind : Optional[List[int]], optional
        List of row numbers representing the samples for which to extract individual
        shapley values. This must be a list of integer indices. The underlying SHAP
        library does not support non-integer indexing.
    background_samples: int,
        Amount of samples to be taken from the dataset in order to build the reference values.
        See documentation about sampling above. Unused if background_kmeans is not False.
    background_kmeans : Union[bool, int], optional
        If True, use SHAP kmeans to create a data summary to serve as background data for the
        SHAP explainer using 50 centroids by default. If an int is provided,
        that will be used as the number of centroids. If False, random sampling will take place.
    """

    def __init__(
        self,
        samples_ind: Optional[List[int]] = None,
        background_samples: int = 100,
        background_kmeans: Union[bool, int] = False,
    ):
        """
        _summary_

        Parameters
        ----------
        samples_ind : Optional[List[int]], optional
            _description_, by default None
        background_samples : int, optional
            _description_, by default 100
        background_kmeans : Union[bool, int], optional
            _description_, by default False
        """

        super().__init__()
        self.samples_ind = samples_ind
        self._validate_samples_ind()
        self.background_samples = background_samples
        self.background_kmeans = background_kmeans

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
        self._setup_shap()
        res = self._get_overall_shap_contributions()
        self.results = [TableContainer(res, **self.get_container_info())]

        ## Sample specific results
        if self.samples_ind:
            labels = {"ordered_feature_names": self.shap_values.feature_names}
            ind_res = self._get_mult_sample_shapley_values()
            self.results += [
                TableContainer(ind_res, **self.get_container_info(labels=labels))
            ]
        return self

    def _setup_shap(self):
        """
        Setup the explainer given the model and the feature dataset
        """
        if self.background_kmeans:
            if type(self.background_kmeans) is int:
                centroids_num = self.background_kmeans
            else:
                centroids_num = 50
            data_summary = kmeans(self.X, centroids_num).data
        else:
            data_summary = self.X.sample(self.background_samples)
        # try to use the model-like, which will only work if it is a model
        # that shap supports
        try:
            explainer = Explainer(self.model.model_like, data_summary)
        except:
            explainer = Explainer(self.model.predict, data_summary)
        # Generating the actual values calling the specific Shap function
        self.shap_values = explainer(self.X)
        return self

    def _get_overall_shap_contributions(self) -> DataFrame:
        """
        Calculate overall SHAP contributions for a dataset.

        The output of SHAP package provides Shapley values for each sample in a
        dataset. To summarize the contribution of each feature in a dataset, the
        samples contributions need to be aggregated.

        For each of the features, this method provides: mean and the
        mean of the absolute value of the samples Shapley values.

        Returns
        -------
        DataFrame
            Summary of the Shapley values across the full dataset.
        """
        values_df = DataFrame(self.shap_values.values)
        values_df.columns = self.shap_values.feature_names
        shap_means = values_df.apply(["mean"]).T
        shap_abs_means = values_df.apply(lambda x: mean(abs(x)))
        shap_abs_means.name = "mean(|x|)"
        final = concat([shap_means, shap_abs_means], axis=1)
        final = final.sort_values("mean(|x|)", ascending=False)
        final.name = "Summary of Shap statistics"

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

    def _validate_samples_ind(self, limit=5):
        """
        Enforce limit on maximum amount of samples for which to extract
        individual shap values.

        A maximum number of samples is enforced, this is in order to constrain the
        amount of information in transit to Credo AI Platform, both for performance
        and security reasons.

        Parameters
        ----------
        limit : int, optional
            Max number of samples allowed, by default 5.

        Raises
        ------
        ValidationError
        """
        if self.samples_ind is not None:
            if len(self.samples_ind) > limit:
                message = "The maximum amount of individual samples_ind allowed is 5."
                raise ValidationError(message)

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
            Contains shapley values for the sample, and the reference value.
            The model prediction for the sample is equal to: ref_value + sum(values)
        """

        return {
            "values": sample_shap.values,
            "ref_value": sample_shap.base_values,
        }