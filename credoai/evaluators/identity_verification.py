import pandas as pd
from credoai.artifacts import ComparisonData, ComparisonModel
from credoai.evaluators import Evaluator
from credoai.evaluators.utils.validation import (
    check_artifact_for_nulls,
    check_data_instance,
    check_model_instance,
    check_existence,
)
from credoai.evidence.containers import MetricContainer

METRIC_SUBSET = [
    "fmr-score",
    "fnmr-score"
]


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

    def __init__(self, thresholds: list = [90], comparison_levels: list = ['sample', 'subject']):
        self.thresholds = thresholds
        self.comparison_levels = comparison_levels

    name = "IdentityVerification"
    required_artifacts = ["model", "assessment_data"]

    def _setup(self):
        self.pairs = self.assessment_data.pairs
        self.subjects_sensitive_features = self.assessment_data.subjects_sensitive_features
        try:
            self.subjects_sensitive_features = self.assessment_data.subjects_sensitive_features
        except:
            self.subjects_sensitive_features = None
        self.results = list()

        return self

    def _validate_arguments(self):
        check_data_instance(self.data, ComparisonData)
        check_model_instance(self.model, ComparisonModel)
        check_artifact_for_nulls(self.assessment_data, "Data")
        check_existence(self.data.pairs, "pairs")
        return self

    def evaluate(self):
        """Runs the assessment process

        Returns
        -------
        dict, nested
            Key: assessment category
            Values: detailed results associated with each category
        """
        # for threshold in self.thresholds:
        #     for level in self.comparison_levels:
        #         df_processed = self._preprocess_data(self.pairs, threshold=threshold, comparison_level=level)
        #         fmr = self._find_fmr(df_processed)
        #         print(fmr)
        #         fnmr = self._find_fnmr(df_processed)
        #         print(fnmr)

        threshold = self.thresholds[0]
        level = self.comparison_levels[0]
        df_processed = self._preprocess_data(self.pairs, threshold=threshold, comparison_level=level)

        fmr_results = fmr = self._find_fmr(df_processed)
        fnmr_results = fmr = self._find_fnmr(df_processed)

        res = {**fmr_results, **fnmr_results}
        res = {k: v for k, v in res.items() if k in METRIC_SUBSET}

        # Reformat results
        res = [pd.DataFrame(v).assign(metric_type=k) for k, v in res.items()]
        res = pd.concat(res)
        res[["type", "subtype"]] = res.metric_type.str.split("-", expand=True)
        res.drop("metric_type", axis=1, inplace=True)

        self.results = [MetricContainer(res, **self.get_container_info())]

        return self

    def _preprocess_data(df, threshold=90, comparison_level='sample'):
        df['match_prediction'] = df.apply(
            lambda x: 1 if x['similarity_score']>=threshold else 0, axis=1
            )
        if comparison_level == 'subject':
            df = df.sort_values('match').drop_duplicates(
                subset=['source-subject-id', 'target-subject-id'], keep='last'
                )
        return df

    def _find_fmr(df):
        n = len(df[df['match']==0])
        fp = len(df[(df['match']==0) & (df['match_prediction']==1)])
        fmr = {"fmr-score": [{"value": fp/n}]}
        return fmr

    def _find_fnmr(df):
        n = len(df[df['match']==1])
        fn = len(df[(df['match']==1) & (df['match_prediction']==0)])
        fnmr = {"fnmr-score": [{"value": fn/n}]}
        return fnmr