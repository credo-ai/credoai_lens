from dis import dis
from typing import List, Union

import pandas as pd
from absl import logging
from credoai.metrics import Metric, find_metrics
from credoai.metrics.metric_constants import MODEL_METRIC_CATEGORIES
from credoai.modules.credo_module import CredoModule
from credoai.modules.model_modules.performance_base import PerformanceModule
from credoai.utils.common import NotRunError, ValidationError, to_array
from fairlearn.metrics import MetricFrame
from scipy.stats import norm
from sklearn.utils import check_consistent_length


class FairnessModule(PerformanceModule):
    """
    Fairness module for Credo AI. Handles any metric that can be
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
    sensitive_features :  pandas.DataFrame
        The segmentation feature(s) which should be used to create subgroups to analyze.
    y_true : (List, pandas.Series, numpy.ndarray)
        The ground-truth labels (for classification) or target values (for regression).
    y_pred : (List, pandas.Series, numpy.ndarray)
        The predicted labels for classification
    y_prob : (List, pandas.Series, numpy.ndarray), optional
        The unthresholded predictions, confidence values or probabilities.
    method : str, optional
        How to compute the differences: "between_groups" or "to_overall".
        See fairlearn.metrics.MetricFrame.difference
        for details, by default 'between_groups'
    """

    def __init__(
        self,
        metrics,
        sensitive_features,
        y_true,
        y_pred,
        y_prob=None,
        method="between_groups",
    ):
        super().__init__(
            metrics=metrics,
            sensitive_features=sensitive_features,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
        )
        # assign variables
        self.fairness_method = method
        self.fairness_metrics = None
        self.fairness_prob_metrics = None
        self.update_metrics(metrics)

    def run(self):
        """
        Run fairness base module

        Returns
        -------
        dict
            Dictionary containing two pandas Dataframes:
                - "disaggregated results": The disaggregated performance metrics, along with acceptability and risk
            as columns
                - "fairness": Dataframe with fairness metrics, along with acceptability and risk
            as columns
        """
        super().run()
        del self.results["overall_performance"]
        fairness_results = self.get_fairness_results()
        self.results["fairness"] = fairness_results
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
        if self.results:
            results = super().prepare_results()
            results = pd.concat([self.results["fairness"], results])
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
            self.fairness_metrics,
            self.fairness_prob_metrics,
            self.failed_metrics,
        ) = self._process_metrics(self.metrics)
        self._setup_metric_frames()

    def get_fairness_results(self):
        """Return fairness and performance parity metrics

        Note, performance parity metrics are labeled with their
        related performance label, but are computed using
        fairlearn.metrics.MetricFrame.difference(method)

        Parameters
        ----------


        Returns
        -------
        pandas.DataFrame
            The returned fairness metrics
        """

        results = []
        for metric_name, metric in self.fairness_metrics.items():
            try:
                metric_value = metric.fun(
                    y_true=self.y_true,
                    y_pred=self.y_pred,
                    sensitive_features=self.sensitive_features,
                    method=self.fairness_method,
                )
            except Exception as e:
                logging.error(
                    f"A metric ({metric_name}) failed to run. "
                    "Are you sure it works with this kind of model and target?\n"
                )
                raise e
            results.append(
                {
                    "metric_type": metric_name,
                    "value": metric_value,
                    "sensitive_feature": self.sensitive_features.name,
                }
            )

        for metric_name, metric in self.fairness_prob_metrics.items():
            try:
                metric_value = metric.fun(
                    y_true=self.y_true,
                    y_prob=self.y_prob,
                    sensitive_features=self.sensitive_features,
                    method=self.fairness_method,
                )
            except Exception as e:
                logging.error(
                    f"A metric ({metric_name}) failed to run. Are you sure it works with this kind of model and target?"
                )
                raise e
            results.append(
                {
                    "metric_type": metric_name,
                    "value": metric_value,
                    "sensitive_feature": self.sensitive_features.name,
                }
            )

        results = pd.DataFrame.from_dict(results)

        # add parity results
        parity_results = pd.Series(dtype=float)
        parity_results = []
        for metric_frame in self.metric_frames.values():
            diffs = metric_frame.difference(method=self.fairness_method)
            diffs = pd.DataFrame({"metric_type": diffs.index, "value": diffs.values})
            diffs["sensitive_feature"] = self.sensitive_features.name
            parity_results.append(diffs)

        if parity_results:
            parity_results = pd.concat(parity_results)
            results = pd.concat([results, parity_results])
        results.set_index("metric_type", inplace=True)
        # add kind
        results["subtype"] = ["fairness"] * len(results)
        results.loc[results.index[-len(parity_results) :], "subtype"] = "parity"
        return results.sort_values(by="sensitive_feature")

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
        fairness_metrics = {}
        fairness_prob_metrics = {}
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
                fairness_metrics[metric_name] = metric
            elif metric.metric_category in MODEL_METRIC_CATEGORIES:
                if metric.takes_prob:
                    prob_metrics[metric_name] = metric
                else:
                    performance_metrics[metric_name] = metric
            else:
                logging.warning(f"{metric_name} failed to be used by FairnessModule")
                failed_metrics.append(metric_name)

        return (
            performance_metrics,
            prob_metrics,
            fairness_metrics,
            fairness_prob_metrics,
            failed_metrics,
        )
