from collections import defaultdict
from typing import List

import pandas as pd
from connect.evidence import MetricContainer, TableContainer

from credoai.artifacts import TabularData
from credoai.evaluators import Evaluator
from credoai.evaluators.utils.fairlearn import setup_metric_frames
from credoai.evaluators.utils.validation import (
    check_artifact_for_nulls,
    check_data_instance,
    check_existence,
)
from credoai.modules.constants_metrics import (
    MODEL_METRIC_CATEGORIES,
    THRESHOLD_METRIC_CATEGORIES,
)
from credoai.modules.metrics import Metric, find_metrics
from credoai.utils.common import ValidationError


class ModelFairness(Evaluator):
    """
    Model Fairness evaluator for Credo AI.

    This evaluator calculates performance metrics disaggregated by a sensitive feature, as
    well as evaluating the parity of those metrics.

    Handles any metric that can be calculated on a set of ground truth labels and predictions,
    e.g., binary classification, multiclass classification, regression.


    Parameters
    ----------
    metrics : List-like
        list of metric names as string or list of Metrics (credoai.metrics.Metric).
        Metric strings should in list returned by credoai.modules.list_metrics.
        Note for performance parity metrics like
        "false negative rate parity" just list "false negative rate". Parity metrics
        are calculated automatically if the performance metric is supplied
    method : str, optional
        How to compute the differences: "between_groups" or "to_overall".
        See fairlearn.metrics.MetricFrame.difference
        for details, by default 'between_groups'
    """

    required_artifacts = {"model", "data", "sensitive_feature"}

    def __init__(
        self,
        metrics=None,
        method="between_groups",
    ):
        self.metrics = metrics
        self.fairness_method = method
        self.fairness_metrics = None
        self.fairness_prob_metrics = None
        super().__init__()

    def _validate_arguments(self):
        check_existence(self.metrics, "metrics")
        check_data_instance(self.data, TabularData)
        check_existence(self.data.sensitive_features, "sensitive_features")
        check_artifact_for_nulls(self.data, "Data")

    def _setup(self):
        self.sensitive_features = self.data.sensitive_feature
        self.y_true = self.data.y
        self.y_pred = self.model.predict(self.data.X)
        if hasattr(self.model, "predict_proba"):
            self.y_prob = self.model.predict_proba(self.data.X)
        else:
            self.y_prob = (None,)
        self.update_metrics(self.metrics)
        self.sens_feat_label = {"sensitive_feature": self.sensitive_features.name}

    def evaluate(self):
        """
        Run fairness base module.
        """
        fairness_results = self.get_fairness_results()
        disaggregated_metrics = self.get_disaggregated_performance()
        disaggregated_thresh_results = self.get_disaggregated_threshold_performance()

        results = []
        for result_obj in [
            fairness_results,
            disaggregated_metrics,
            disaggregated_thresh_results,
        ]:
            if result_obj is not None:
                try:
                    results += result_obj
                except TypeError:
                    results.append(result_obj)

        self.results = results
        return self

    def update_metrics(self, metrics, replace=True):
        """
        Replace metrics

        Parameters
        ----------
        metrics : List-like
            list of metric names as string or list of Metrics (credoai.metrics.Metric).
            Metric strings should in list returned by credoai.modules.list_metrics.
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
            self.threshold_metrics,
            self.fairness_metrics,
            self.failed_metrics,
        ) = self._process_metrics(self.metrics)
        self.metric_frames = setup_metric_frames(
            self.performance_metrics,
            self.prob_metrics,
            self.threshold_metrics,
            self.y_pred,
            self.y_prob,
            self.y_true,
            self.sensitive_features,
        )

    def get_disaggregated_performance(self):
        """
        Return performance metrics for each group

        Parameters
        ----------
        melt : bool, optional
            If True, return a long-form dataframe, by default False

        Returns
        -------
        TableContainer
            The disaggregated performance metrics
        """
        disaggregated_df = pd.DataFrame()
        for name, metric_frame in self.metric_frames.items():
            if name == "thresh":
                continue
            df = metric_frame.by_group.copy().convert_dtypes()
            disaggregated_df = pd.concat([disaggregated_df, df], axis=1)

        if disaggregated_df.empty:
            self.logger.warn("No disaggregated metrics could be calculated.")
            return

        # reshape
        disaggregated_results = disaggregated_df.reset_index().melt(
            id_vars=[disaggregated_df.index.name],
            var_name="type",
        )
        disaggregated_results.name = "disaggregated_performance"

        metric_type_label = {
            "metric_types": disaggregated_results.type.unique().tolist()
        }

        return TableContainer(
            disaggregated_results,
            **self.get_container_info(
                labels={**self.sens_feat_label, **metric_type_label}
            ),
        )

    def get_disaggregated_threshold_performance(self):
        """
        Return performance metrics for each group

        Parameters
        ----------
        melt : bool, optional
            If True, return a long-form dataframe, by default False

        Returns
        -------
        List[TableContainer]
            The disaggregated performance metrics
        """
        metric_frame = self.metric_frames.get("thresh")
        if metric_frame is None:
            return
        df = metric_frame.by_group.copy().convert_dtypes()

        df = df.reset_index().melt(
            id_vars=[df.index.name],
            var_name="type",
        )

        to_return = defaultdict(list)
        for i, row in df.iterrows():
            tmp_df = row["value"]
            tmp_df = tmp_df.assign(**row.drop("value"))
            to_return[row["type"]].append(tmp_df)
        for key in to_return.keys():
            df = pd.concat(to_return[key])
            df.name = "threshold_dependent_disaggregated_performance"
            to_return[key] = df

        disaggregated_thresh_results = []
        for key, df in to_return.items():
            labels = {**self.sens_feat_label, **{"metric_type": key}}
            disaggregated_thresh_results.append(
                TableContainer(df, **self.get_container_info(labels=labels))
            )

        return disaggregated_thresh_results

    def get_fairness_results(self):
        """Return fairness and performance parity metrics

        Note, performance parity metrics are labeled with their
        related performance label, but are computed using
        fairlearn.metrics.MetricFrame.difference(method)

        Returns
        -------
        MetricContainer
            The returned fairness metrics
        """

        results = []
        for metric_name, metric in self.fairness_metrics.items():
            pred_argument = {"y_pred": self.y_pred}
            if metric.takes_prob:
                pred_argument = {"y_prob": self.y_prob}
            try:
                metric_value = metric.fun(
                    y_true=self.y_true,
                    sensitive_features=self.sensitive_features,
                    method=self.fairness_method,
                    **pred_argument,
                )
            except Exception as e:
                self.logger.error(
                    f"A metric ({metric_name}) failed to run. "
                    "Are you sure it works with this kind of model and target?\n"
                )
                raise e
            results.append({"metric_type": metric_name, "value": metric_value})

        results = pd.DataFrame.from_dict(results)

        # add parity results
        parity_results = pd.Series(dtype=float)
        parity_results = []
        for name, metric_frame in self.metric_frames.items():
            if name == "thresh":
                # Don't calculate difference for curve metrics. This is not mathematically well-defined.
                continue
            diffs = metric_frame.difference(self.fairness_method).rename(
                "{}_parity".format
            )
            diffs = pd.DataFrame({"metric_type": diffs.index, "value": diffs.values})
            parity_results.append(diffs)

        if parity_results:
            parity_results = pd.concat(parity_results)
            results = pd.concat([results, parity_results])

        results.rename({"metric_type": "type"}, axis=1, inplace=True)

        if results.empty:
            self.logger.info("No fairness metrics calculated.")
            return
        return MetricContainer(
            results,
            **self.get_container_info(labels=self.sens_feat_label),
        )

    def _process_metrics(self, metrics):
        """
        Separates metrics

        Parameters
        ----------
        metrics : Union[List[Metric, str]]
            list of metrics to use. These can be Metric objects (see credoai.modules.metrics.py), or strings.
            If strings, they will be converted to Metric objects using find_metrics

        Returns
        -------
        Separate dictionaries and lists of metrics
        """
        # separate metrics
        failed_metrics = []
        performance_metrics = {}
        prob_metrics = {}
        threshold_metrics = {}
        fairness_metrics = {}
        fairness_prob_metrics = {}
        for metric in metrics:
            if isinstance(metric, str):
                metric_name = metric
                metric_categories_to_include = MODEL_METRIC_CATEGORIES
                metric_categories_to_include.append(self.model.type)
                metric = find_metrics(metric, metric_categories_to_include)
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
                    if metric.metric_category in THRESHOLD_METRIC_CATEGORIES:
                        threshold_metrics[metric_name] = metric
                    else:
                        prob_metrics[metric_name] = metric
                else:
                    performance_metrics[metric_name] = metric
            else:
                self.logger.warning(
                    f"{metric_name} failed to be used by FairnessModule"
                )
                failed_metrics.append(metric_name)

        return (
            performance_metrics,
            prob_metrics,
            threshold_metrics,
            fairness_metrics,
            failed_metrics,
        )
