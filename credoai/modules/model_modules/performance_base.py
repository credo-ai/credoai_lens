from collections import defaultdict
from typing import List, Union

import pandas as pd
from absl import logging
from credoai.metrics import Metric, find_metrics
from credoai.metrics.metric_constants import MODEL_METRIC_CATEGORIES
from credoai.modules.credo_module import CredoModule
from credoai.utils.common import NotRunError, ValidationError, to_array
from fairlearn.metrics import MetricFrame
from scipy.stats import norm
from sklearn.utils import check_consistent_length


class PerformanceModule(CredoModule):
    """
    Performance module for Credo AI. Handles any metric that can be
    calculated on a set of ground truth labels and predictions,
    e.g., binary classification, multiclass classification, regression.

    This module takes in a set of metrics  and provides functionality to:
    - calculate the metrics
    - create disaggregated metrics

    Parameters
    ----------
    metrics : List-like
        list of metric names as string or list of Metrics (credoai.metrics.Metric).
        Metric strings should in list returned by credoai.metrics.list_metrics.
        Note for performance parity metrics like
        "false negative rate parity" just list "false negative rate". Parity metrics
        are calculated automatically if the performance metric is supplied
    y_true : (List, pandas.Series, numpy.ndarray)
        The ground-truth labels (for classification) or target values (for regression).
    y_pred : (List, pandas.Series, numpy.ndarray)
        The predicted labels for classification
    y_prob : (List, pandas.Series, numpy.ndarray), optional
        The unthresholded predictions, confidence values or probabilities.
    sensitive_features :  pandas.Series
        The segmentation feature which should be used to create subgroups to analyze.
    """

    def __init__(self, metrics, y_true, y_pred, y_prob=None, sensitive_features=None):
        super().__init__()
        # data variables
        self.y_true = to_array(y_true)
        self.y_pred = to_array(y_pred)
        self.y_prob = to_array(y_prob) if y_prob is not None else None
        self.perform_disaggregation = True
        if sensitive_features is None:
            self.perform_disaggregation = False
            # only set to use metric frame
            sensitive_features = pd.Series(["NA"] * len(self.y_true), name="NA")
        self.sensitive_features = sensitive_features
        self._validate_inputs()

        # assign variables
        self.metrics = metrics
        self.metric_frames = {}
        self.performance_metrics = None
        self.prob_metrics = None
        self.failed_metrics = None
        self.update_metrics(metrics)

    def run(self):
        """
        Run performance base module


        Returns
        -------
        self
        """
        self.results = {"overall_performance": self.get_overall_metrics()}
        if self.perform_disaggregation:
            self.results.update(self.get_disaggregated_performance())
        return self

    def prepare_results(self):
        """Prepares results for Credo AI's governance platform

        Structures results for export as a dataframe with appropriate structure
        for exporting. See credoai.modules.credo_module.

        Returns
        -------
        pd.DataFrame
        Raises
        ------
        NotRunError
            Occurs if self.run is not called yet to generate the raw assessment results
        """
        if self.results is not None:
            if "overall_performance" in self.results:
                results = self.results["overall_performance"]
            else:
                results = pd.DataFrame()

            if self.perform_disaggregation:
                disaggregated_df = self.results[f"disaggregated_performance"].copy()
                disaggregated_df = (
                    disaggregated_df.reset_index()
                    .melt(
                        id_vars=[disaggregated_df.index.name, "subtype"],
                        var_name="metric_type",
                    )
                    .set_index("metric_type")
                )
                disaggregated_df["sensitive_feature"] = self.sensitive_features.name
                results = pd.concat([results, disaggregated_df])
            return results
        else:
            raise NotRunError(
                "Results not created yet. Call 'run' with appropriate arguments before preparing results"
            )

    def update_metrics(self, metrics, replace=True):
        """replace metrics

        Parameters
        ----------
        metrics : List-like
            list of metric names as string or list of Metrics (credoai.metrics.Metric).
            Metric strings should in list returned by credoai.metrics.list_metrics.
            Note for performance parity metrics like
            "false negative rate parity" just list "false negative rate". Parity metrics
            are calculated automatically if the performance metric is supplied
        """
        if replace:
            self.metrics = metrics
        else:
            self.metrics += metrics
        (
            self.performance_metrics,
            self.prob_metrics,
            self.failed_metrics,
        ) = self._process_metrics(self.metrics)
        self._setup_metric_frames()

    def get_df(self):
        """Return dataframe of input arrays

        Returns
        -------
        pandas.DataFrame
            Dataframe containing the input arrays
        """
        df = pd.DataFrame({"true": self.y_true, "pred": self.y_pred})
        if self.sensitive_features.name != "NA":
            df = df.join(self.sensitive_features.reset_index(drop=True))
        if self.y_prob is not None:
            y_prob_df = pd.DataFrame(self.y_prob)
            y_prob_df.columns = [f"y_prob_{i}" for i in range(y_prob_df.shape[1])]
            df = pd.concat([df, y_prob_df], axis=1)

        return df

    def get_overall_metrics(self):
        """Return performance metrics for each group

        Returns
        -------
        pandas.Series
            The overall performance metrics
        """
        # retrive overall metrics for one of the sensitive features only as they are the same
        overall_metrics = [
            metric_frame.overall for metric_frame in self.metric_frames.values()
        ]
        if overall_metrics:
            output_series = (
                pd.concat(overall_metrics, axis=0)
                .rename(index="value")
                .to_frame()
                .assign(subtype="overall_performance")
            )
            return output_series
        else:
            logging.warn("No overall metrics could be calculated.")

    def get_disaggregated_performance(self):
        """Return performance metrics for each group

        Parameters
        ----------
        melt : bool, optional
            If True, return a long-form dataframe, by default False

        Returns
        -------
        pandas.DataFrame
            The disaggregated performance metrics
        """
        disaggregated_results = {}
        disaggregated_df = pd.DataFrame()
        for metric_frame in self.metric_frames.values():
            df = metric_frame.by_group.copy().convert_dtypes()
            disaggregated_df = pd.concat([disaggregated_df, df], axis=1)
        disaggregated_results["disaggregated_performance"] = disaggregated_df.assign(
            subtype="disaggregated_performance"
        )
        if not disaggregated_results:
            logging.warn("No disaggregated metrics could be calculated.")
        return disaggregated_results

    def get_sensitive_feature(self):
        if self.sensitive_features.name != "NA":
            return self.sensitive_features

    def _process_metrics(self, metrics):
        """Separates metrics

        Parameters
        ----------
        metrics : Union[List[Metirc, str]]
            list of metrics to use. These can be Metric objects (credoai.metric.metrics) or
            strings. If strings, they will be converted to Metric objects using find_metrics

        Returns
        -------
        Separate dictionaries and lists of metrics
        """
        # separate metrics
        failed_metrics = []
        performance_metrics = {}
        prob_metrics = {}
        for metric in metrics:
            if isinstance(metric, str):
                metric_name = metric
                metric = find_metrics(metric, MODEL_METRIC_CATEGORIES)
                if len(metric) == 1:
                    metric = metric[0]
                elif len(metric) == 0:
                    raise Exception(
                        f"Returned no metrics when searching using the provided metric name <{metric_name}>. Expected to find one matching metric."
                    )
                else:
                    raise Exception(
                        f"Returned multiple metrics when searching using the provided metric name <{metric_name}>. Expected to find only one matching metric."
                    )
            else:
                metric_name = metric.name
            if not isinstance(metric, Metric):
                raise ValidationError("Metric is not of type credoai.metric.Metric")
            if metric.metric_category == "FAIRNESS":
                logging.info(
                    f"fairness metric, {metric_name}, unused by PerformanceModule"
                )
                pass
            elif metric.metric_category in MODEL_METRIC_CATEGORIES:
                if metric.takes_prob:
                    prob_metrics[metric_name] = metric
                else:
                    performance_metrics[metric_name] = metric
            else:
                logging.warning(f"{metric_name} failed to be used by FairnessModule")
                failed_metrics.append(metric_name)

        return (performance_metrics, prob_metrics, failed_metrics)

    def _create_metric_frame(self, metrics, y_pred, sensitive_features):
        """Creates metric frame from dictionary of key:Metric"""
        metrics = {name: metric.fun for name, metric in metrics.items()}
        return MetricFrame(
            metrics=metrics,
            y_true=self.y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )

    def _setup_metric_frames(self):
        self.metric_frames = {}
        if self.y_pred is not None and self.performance_metrics:
            self.metric_frames["pred"] = self._create_metric_frame(
                self.performance_metrics,
                self.y_pred,
                sensitive_features=self.sensitive_features,
            )

            # for metrics that require the probabilities
            self.prob_metric_frame = None
            if self.y_prob is not None and self.prob_metrics:
                self.metric_frames["prob"] = self._create_metric_frame(
                    self.prob_metrics,
                    self.y_prob,
                    sensitive_features=self.sensitive_features,
                )

    def _validate_inputs(self):
        check_consistent_length(
            self.y_true, self.y_pred, self.y_prob, self.sensitive_features
        )
