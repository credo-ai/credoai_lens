"""Identity Verification evaluator"""
import pandas as pd
from credoai.artifacts import ComparisonData, ComparisonModel
from credoai.evaluators import Evaluator
from credoai.evaluators.utils.fairlearn import setup_metric_frames
from credoai.evaluators.utils.validation import (check_data_instance,
                                                 check_existence,
                                                 check_model_instance)
from credoai.evidence.containers import MetricContainer, TableContainer
from credoai.modules.metric_constants import \
    BINARY_CLASSIFICATION_FUNCTIONS as bcf
from credoai.modules.metrics import Metric

METRIC_SUBSET = [
    'false_match_rate-score',
    'false_non_match_rate-score'
    ]


class IdentityVerification(Evaluator):
    """Pair-wise-comparison-based identity verification evaluator for Credo AI

    This evaluator takes in identity verification data and
        provides functionality to perform performance and fairness assessment

    Parameters
    ----------
    pairs : pd.DataFrame of shape (n_pairs, 4)
        Dataframe where each row represents a data sample pair and associated subjects
        Type of data sample is decided by the ComparisonModel's `compare` function, which takes
        data sample pairs and returns their similarity scores. Examples are selfies, fingerprint scans,
        or voices of a person.
        Required columns:
            source-subject-id: unique identifier of the source subject
            source-subject-data-sample: data sample from the source subject
            target-subject-id: unique identifier of the target subject
            target-subject-data-sample: data sample from the target subject
    subjects_sensitive_features : pd.DataFrame of shape (n_subjects, n_sensitive_feature_names), optional
        Sensitive features of all subjects present in pairs dataframe
        If provided, disaggregated performance assessment is also performed.
        This can be the columns you want to perform segmentation analysis on, or
        a feature related to fairness like 'race' or 'gender'
        Required columns:
            subject-id: id of subjects. Must cover all the subjects inlcluded in `pairs` dataframe
            other columns with arbitrary names for sensitive features
    similarity_thresholds : list
        list of similarity score thresholds
        Similarity equal or greater than a similarity score threshold means match
    comparison_levels : list
        list of comparison levels. Options:
            sample: it means a match is observed for every sample pair. Sample-level comparison represent
                a use case where only two samples (such as a real time selfie and stored ID image) are
                used to confirm an identity.
            subject: it means if any pairs of samples for the same subject are a match, the subject pair
                is marked as a match. Some identity verification use cases improve overall accuracy by storing
                multiple samples per identity. Subject-level comparison mirrors this behavior.
    """

    def __init__(
        self,
        similarity_thresholds: list = [90, 95],
        comparison_levels: list = ["sample", "subject"],
    ):
        self.similarity_thresholds = similarity_thresholds
        self.comparison_levels = comparison_levels
        super().__init__()

    required_artifacts = {"model", "assessment_data"}

    def _setup(self):
        self.pairs = self.assessment_data.pairs
        try:
            self.subjects_sensitive_features = (
                self.assessment_data.subjects_sensitive_features
            )
            sensitive_features_names = list(self.subjects_sensitive_features.columns)
            sensitive_features_names.remove("subject-id")
            self.sensitive_features_names = sensitive_features_names
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

        self.results = self._assess_overall_performance()

        if self.subjects_sensitive_features is not None:
            self._assess_disaggregated_performance()

        return self

    def _process_data(
        self, pairs_processed, threshold=90, comparison_level="sample", sf=None
    ):
        """Process the pairs and sensitive features dataframes

        Parameters
        ----------
        pairs_processed : pd.DataFrame
            pairs dataframe to be processed in place
        threshold : float, optional
            similarity threshold equal or greater than which mean match, by default 90
        comparison_level : str, optional
            comparison levels, by default "sample"
            Options:
                sample: it means a match is observed for every sample pair. Sample-level comparison represent
                    a use case where only two samples (such as a real time selfie and stored ID image) are
                    used to confirm an identity.
                subject: it means if any pairs of samples for the same subject are a match, the subject pair
                    is marked as a match. Some identity verification use cases improve overall accuracy by storing
                    multiple samples per identity. Subject-level comparison mirrors this behavior.
        sf : pd.DataFrame, optional
            sensitive feature dataframe with 'subject-id' and sensitive feature name columns, by default None

        Returns
        -------
        pd.DataFrame, pd.DataFrame
            processeded pairs and sensitive features dataframes
        """
        pairs_processed["match_prediction"] = pairs_processed.apply(
            lambda x: 1 if x["similarity_score"] >= threshold else 0, axis=1
        )
        if comparison_level == "subject":
            pairs_processed = pairs_processed.sort_values("match").drop_duplicates(
                subset=["source-subject-id", "target-subject-id"], keep="last"
            )

        sf_processed = None
        if sf is not None:
            # Process the data for disaggregated assessment
            #  Filter out the pairs with non-matching sensitive feature groups
            #  and create the sensitive feature vector
            sf_name = list(sf.columns)
            sf_name.remove("subject-id")
            sf_name = sf_name[0]
            pairs_processed = pairs_processed.merge(
                sf, left_on="source-subject-id", right_on="subject-id", how="left"
            )
            pairs_processed.drop("subject-id", inplace=True, axis=1)
            pairs_processed.rename(
                {sf_name: sf_name + "-source-subject"}, inplace=True, axis=1
            )
            pairs_processed = pairs_processed.merge(
                sf, left_on="target-subject-id", right_on="subject-id", how="left"
            )
            pairs_processed.drop("subject-id", inplace=True, axis=1)
            pairs_processed = pairs_processed.loc[
                pairs_processed[sf_name + "-source-subject"] == pairs_processed[sf_name]
            ]
            sf_processed = pairs_processed[sf_name]
            pairs_processed.drop(
                [sf_name, sf_name + "-source-subject"], inplace=True, axis=1
            )

        return pairs_processed, sf_processed

    def _assess_overall_performance(self):
        """Perform overall performance assessment"""
        overall_performance_res = []
        for threshold in self.similarity_thresholds:
            for level in self.comparison_levels:
                cols = ["subject-id", "gender"]
                sf = self.subjects_sensitive_features[cols]
                pairs_processed, sf_processed = self._process_data(
                    self.pairs.copy(),
                    threshold=threshold,
                    comparison_level=level,
                    sf=sf,
                )

                fmr = bcf["false_positive_rate"](
                    pairs_processed["match"], pairs_processed["match_prediction"]
                )
                fmr_results = {"false_match_rate-score": [{"value": fmr}]}

                fnmr = bcf["false_negative_rate"](
                    pairs_processed["match"], pairs_processed["match_prediction"]
                )
                fnmr_results = {"false_non_match_rate-score": [{"value": fnmr}]}

                res = {**fmr_results, **fnmr_results}
                res = {k: v for k, v in res.items() if k in METRIC_SUBSET}

                res = [pd.DataFrame(v).assign(metric_type=k) for k, v in res.items()]
                res = pd.concat(res)

                res[["type", "subtype"]] = res.metric_type.str.split("-", expand=True)
                res.drop("metric_type", axis=1, inplace=True)
                parameters_label = {
                    "similarity_threshold": threshold,
                    "comparison_level": level,
                }
                overall_performance_res.append(
                    MetricContainer(
                        res, **self.get_container_info(labels={**parameters_label})
                    )
                )
            
        return overall_performance_res

    def _assess_disaggregated_performance(self):
        """Perform disaggregated performance assessment"""
        performance_metrics = {
            "false_match_rate": Metric(
                "false_match_rate", "BINARY_CLASSIFICATION", bcf["false_positive_rate"]
            ),
            "false_non_match_rate": Metric(
                "false_non_match_rate",
                "BINARY_CLASSIFICATION",
                bcf["false_negative_rate"],
            ),
        }
        for sf_name in self.sensitive_features_names:
            for threshold in self.similarity_thresholds:
                for level in self.comparison_levels:
                    self._assess_disaggregated_performance_one(
                        sf_name, threshold, level, performance_metrics
                    )

    def _assess_disaggregated_performance_one(
        self, sf_name, threshold, level, performance_metrics
    ):
        """Perform disaggregated performance assessment for one combination 
        
        One combination of similarity threshold, comparision level, and sensitive feature

        Parameters
        ----------
        sf_name : str
            sesnsitive feature name
        threshold : float
            similarity threshold
        level : str
            comparison level
        performance_metrics : dict
            performance metrics
        """
        cols = ["subject-id", sf_name]
        sf = self.subjects_sensitive_features[cols]
        pairs_processed, sf_processed = self._process_data(
            self.pairs.copy(),
            threshold=threshold,
            comparison_level=level,
            sf=sf,
        )

        self.metric_frames = setup_metric_frames(
            performance_metrics,
            prob_metrics=None,
            thresh_metrics=None,
            y_pred=pairs_processed["match_prediction"],
            y_prob=None,
            y_true=pairs_processed["match"],
            sensitive_features=sf_processed,
        )

        disaggregated_df = pd.DataFrame()
        for name, metric_frame in self.metric_frames.items():
            df = metric_frame.by_group.copy().convert_dtypes()
            disaggregated_df = pd.concat([disaggregated_df, df], axis=1)
        disaggregated_results = disaggregated_df.reset_index().melt(
            id_vars=[disaggregated_df.index.name],
            var_name="type",
        )
        disaggregated_results.name = "disaggregated_performance"

        sens_feat_label = {"sensitive_feature": sf_name}
        metric_type_label = {
            "metric_types": disaggregated_results.type.unique().tolist()
        }
        parameters_label = {
            "similarity_threshold": threshold,
            "comparison_level": level,
        }
        if disaggregated_results is not None:
            e = TableContainer(
                disaggregated_results,
                **self.get_container_info(
                    labels={
                        **sens_feat_label,
                        **metric_type_label,
                        **parameters_label,
                    }
                ),
            )
            self._results.append(e)
