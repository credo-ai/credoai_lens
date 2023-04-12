import enum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from connect.evidence import TableContainer
from shap import Explainer, Explanation, kmeans

from credoai.evaluators.evaluator import Evaluator
from credoai.evaluators.utils.validation import check_requirements_existence
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

    Required Artifacts
    ------------------
        **Required Artifacts**

        Generally artifacts are passed directly to :class:`credoai.lens.Lens`, which
        handles evaluator setup. However, if you are using the evaluator directly, you
        will need to pass the following artifacts when instantiating the evaluator:

        - model: :class:`credoai.artifacts.Model`
        - assessment_data: :class:`credoai.artifacts.TabularData`

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

    required_artifacts = ["model", "assessment_data"]

    def __init__(
        self,
        samples_ind: Optional[List[int]] = None,
        background_samples: int = 100,
        background_kmeans: Union[bool, int] = False,
    ):
        super().__init__()
        self.samples_ind = samples_ind
        self._validate_samples_ind()
        self.background_samples = background_samples
        self.background_kmeans = background_kmeans
        self.classes = [None]

    def _validate_arguments(self):
        check_requirements_existence(self)

    def _setup(self):
        self.X = self.assessment_data.X
        self.model = self.model
        return self

    def evaluate(self):
        ## Overall stats
        self._setup_shap()
        self.results = [
            TableContainer(self._get_overall_shap_contributions(), **self.get_info())
        ]

        ## Sample specific results
        if self.samples_ind:
            ind_res = self._get_mult_sample_shapley_values()
            self.results += [TableContainer(ind_res, **self.get_info())]
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

        # Define values dataframes and classes variables depending on
        # the shape of the returned values. This accounts for multi class
        # classification
        s_values = self.shap_values.values
        if len(s_values.shape) == 2:
            self.values_df = [pd.DataFrame(s_values)]
        elif len(s_values.shape) == 3:
            self.values_df = [
                pd.DataFrame(s_values[:, :, i]) for i in range(s_values.shape[2])
            ]
            self.classes = self.model.model_like.classes_
        else:
            raise RuntimeError(
                f"Shap vales have unsupported format. Detected shape {s_values.shape}"
            )
        return self

    def _get_overall_shap_contributions(self) -> pd.DataFrame:
        """
        Calculate overall SHAP contributions for a dataset.

        The output of SHAP package provides Shapley values for each sample in a
        dataset. To summarize the contribution of each feature in a dataset, the
        samples contributions need to be aggregated.

        For each of the features, this method provides: mean and the
        mean of the absolute value of the samples Shapley values.

        Returns
        -------
        pd.DataFrame
            Summary of the Shapley values across the full dataset.
        """

        shap_summaries = [
            self._summarize_shap_values(frame) for frame in self.values_df
        ]
        if len(self.classes) > 1:
            for label, df in zip(self.classes, shap_summaries):
                df.assign(class_label=label)
        # fmt: off
        shap_summary = (
            pd.concat(shap_summaries)
            .reset_index()
            .rename({"index": "feature_name"}, axis=1)
        )
        # fmt: on
        shap_summary.name = "Summary of Shap statistics"
        return shap_summary

    def _summarize_shap_values(self, shap_val: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize Shape values at a Dataset level.

        Parameters
        ----------
        shap_val : pd.DataFrame
            Table containing shap values, if the model output is multiclass,
            the table corresponds to the values for a single class.

        Returns
        -------
        pd.DataFrame
            Summarized shap values.
        """
        shap_val.columns = self.shap_values.feature_names
        summaries = {"mean": np.mean, "mean(|x|)": lambda x: np.mean(np.abs(x))}
        results = map(lambda func: shap_val.apply(func), summaries.values())
        # fmt: off
        final = (
            pd.concat(results, axis=1) 
            .set_axis(summaries.keys(), axis=1)
            .sort_values("mean(|x|)", ascending=False)
        )
        # fmt: on
        final.name = "Summary of Shap statistics"
        return final

    def _get_mult_sample_shapley_values(self) -> pd.DataFrame:
        """
        Return shapley values for multiple samples from the dataset.

        Returns
        -------
        pd.DataFrame
            Columns:
                values -> shap values
                ref_value -> Reference value for the shap values
                    (generally the same across the dataset)
                sample_pos -> Position of the sample in the dataset
        """
        all_sample_shaps = []
        for ind in self.samples_ind:
            sample_results = self._get_single_sample_values(self.shap_values[ind])
            sample_results = sample_results.assign(sample_pos=ind)
            all_sample_shaps.append(sample_results)

        res = pd.concat(all_sample_shaps)
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

    def _get_single_sample_values(self, sample_shap: Explanation) -> pd.DataFrame:
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

        class_values = []

        if len(self.classes) == 1:
            return pd.DataFrame({"values": sample_shap.values}).assign(
                ref_value=sample_shap.base_values,
                column_names=self.shap_values.feature_names,
            )

        for label, cls in enumerate(self.classes):
            class_values.append(
                pd.DataFrame({"values": sample_shap.values[:, label]}).assign(
                    class_label=cls,
                    ref_value=sample_shap.base_values[label],
                    column_names=self.shap_values.feature_names,
                )
            )
        return pd.concat(class_values)
