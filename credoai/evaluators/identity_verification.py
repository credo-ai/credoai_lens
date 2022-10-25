import pandas as pd
from credoai.artifacts import ComparisonData, ComparisonModel
from credoai.evaluators import Evaluator
from credoai.evaluators.utils.validation import (
    check_data_instance,
    check_model_instance,
    check_existence,
)
from credoai.evidence.containers import MetricContainer
from credoai.modules.metric_constants import BINARY_CLASSIFICATION_FUNCTIONS as bcf

METRIC_SUBSET = ["false_match_rate-score", "false_non_match_rate-score"]


class IdentityVerification(Evaluator):
    """Pair-wise-comparison-based identity verification evaluator for Credo AI

    This evaluator takes in identity verification data and
        provides functionality to perform performance and fairness assessment

    Parameters
    ----------
    pairs : pd.DataFrame of shape (n_pairs, 4)
        Dataframe where each row represents a data sample pair and associated subjects
        Required columns:
            source-subject-id: unique identifier of the source subject
            source-subject-data-sample: data sample from the source subject
            target-subject-id: unique identifier of the target subject
            target-subject-data-sample: data sample from the target subject
    subjects_sensitive_features : pd.DataFrame of shape (n_subjects, n_sensitive_feature_names), optional
        Sensitive features of all subjects present in pairs dataframe
        This will be used for disaggregating performance
        metrics. This can be the columns you want to perform segmentation analysis on, or
        a feature related to fairness like 'race' or 'gender'
        Required columns:
            subject-id: id of subjects
            other columns with arbitrary names for sensitive features
    thresholds : list
        list of similarity score thresholds
    comparison_levels : list
        list of comparison_levels. Options:
            sample: it means a match is observed for every sample pair. Sample-level comparison represent
                a use case where only two samples (such as a real time selfie and stored ID image) are
                used to confirm an identity.
            subject: it means if any pairs of samples for the same subject are a match, the subject pair
                is marked as a match. Some identity verification use cases improve overall accuracy by storing
                multiple samples per identity. Subject-level comparison mirrors this behavior.
    """

    def __init__(
        self, thresholds: list = [90, 95], comparison_levels: list = ["sample", "subject"]
    ):
        self.thresholds = thresholds
        self.comparison_levels = comparison_levels

    name = "IdentityVerification"
    required_artifacts = ["model", "assessment_data"]

    def _setup(self):
        self.pairs = self.assessment_data.pairs
        try:
            self.subjects_sensitive_features = (
                self.assessment_data.subjects_sensitive_features
            )
        except:
            self.subjects_sensitive_features = None

        self.pairs["similarity_score"] = self.model.compare(
            [
                list(pair)
                for pair in zip(
                    self.pairs["source-subject-data-sample"].tolist(),
                    self.pairs["target-subject-data-sample"].tolist(),
                )
            ]
        )

        self.pairs["match"] = self.pairs.apply(
            lambda x: 1 if x["source-subject-id"] == x["target-subject-id"] else 0,
            axis=1,
        )

        self.results = list()

        return self

    def _validate_arguments(self):
        check_data_instance(self.assessment_data, ComparisonData)
        check_model_instance(self.model, ComparisonModel)
        check_existence(self.assessment_data.pairs, "pairs")
        return self

    def evaluate(self):
        """Runs the assessment process

        Returns
        -------
        dict, nested
            Key: assessment category
            Values: detailed results associated with each category
        """

        res_list = []
        for threshold in self.thresholds:
            for level in self.comparison_levels:
                df_processed = self._preprocess_data(
                    self.pairs, threshold=threshold, comparison_level=level
                )

                fmr = bcf["false_positive_rate"](
                    df_processed["match"], df_processed["match_prediction"]
                )
                fmr_results = {
                    "false_match_rate-score": [{"value": fmr}]
                }

                fnmr = bcf["false_negative_rate"](
                    df_processed["match"], df_processed["match_prediction"]
                )
                fnmr_results = {
                    "false_non_match_rate-score": [{"value": fnmr}]
                } 

                res = {**fmr_results, **fnmr_results}
                res = {k: v for k, v in res.items() if k in METRIC_SUBSET}

                res = [pd.DataFrame(v).assign(metric_type=k) for k, v in res.items()]
                res = pd.concat(res)
                
                res['threshold'] = threshold
                res['comparison_level'] = level
                res_list.append(res)

        res_all = pd.concat(res_list)
        res_all[["type", "subtype"]] = res_all.metric_type.str.split("-", expand=True)
        res_all.drop("metric_type", axis=1, inplace=True)

        self._results = [MetricContainer(res_all, **self.get_container_info(labels={"sensitive_feature": 'sensitive_feature'}))]

        return self

    def _preprocess_data(self, df, threshold=90, comparison_level="sample"):
        df["match_prediction"] = df.apply(
            lambda x: 1 if x["similarity_score"] >= threshold else 0, axis=1
        )
        if comparison_level == "subject":
            df = df.sort_values("match").drop_duplicates(
                subset=["source-subject-id", "target-subject-id"], keep="last"
            )
        return df
