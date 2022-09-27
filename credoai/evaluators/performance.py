import pandas as pd
from credoai.artifacts import TabularData
from credoai.evaluators import Evaluator
from credoai.evaluators.utils.fairlearn import setup_metric_frames
from credoai.evaluators.utils.validation import (
    check_artifact_for_nulls,
    check_data_instance,
    check_existence,
)
from credoai.evidence import MetricContainer
from credoai.modules.metric_constants import MODEL_METRIC_CATEGORIES
from credoai.modules.metrics import Metric, find_metrics
from credoai.utils import global_logger
from credoai.utils.common import ValidationError
from this import d


class Performance(Evaluator):
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
    """

    name = "Performance"
    required_artifacts = {"model", "assessment_data"}

    def __init__(self, metrics=None):
        super().__init__()
        # assign variables
        self.metrics = metrics
        self.metric_frames = {}
        self.performance_metrics = None
        self.prob_metrics = None
        self.failed_metrics = None
        self.perform_disaggregation = True

    def _setup(self):
        # data variables
        self.y_true = self.assessment_data.y
        self.y_pred = self.model.predict(self.assessment_data.X)
        try:
            self.y_prob = self.model.predict_proba(self.assessment_data.X)
        except:
            self.y_prob = None
        self.update_metrics(self.metrics)

        return self

    def evaluate(self):
        """
        Run performance base module


        Returns
        -------
        self
        """
        self._prepare_results()
        return self

    def _prepare_results(self):
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
        self.results = [
            MetricContainer(self.get_overall_metrics(), **self.get_container_info())
        ]

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

        dummy_sensitive = pd.Series(["NA"] * len(self.y_true), name="NA")
        self.metric_frames = setup_metric_frames(
            self.performance_metrics,
            self.prob_metrics,
            self.y_pred,
            self.y_prob,
            self.y_true,
            dummy_sensitive,
        )

    def get_df(self):
        """Return dataframe of input arrays

        Returns
        -------
        pandas.DataFrame
            Dataframe containing the input arrays
        """
        df = pd.DataFrame({"true": self.y_true, "pred": self.y_pred})
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
                pd.concat(overall_metrics, axis=0).rename(index="value").to_frame()
            )
            return output_series.reset_index().rename({"index": "type"}, axis=1)
        else:
            global_logger.warn("No overall metrics could be calculated.")

    @staticmethod
    def _process_metrics(metrics):
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
                global_logger.info(
                    f"fairness metric, {metric_name}, unused by PerformanceModule"
                )
                pass
            elif metric.metric_category in MODEL_METRIC_CATEGORIES:
                if metric.takes_prob:
                    prob_metrics[metric_name] = metric
                else:
                    performance_metrics[metric_name] = metric
            else:
                global_logger.warning(
                    f"{metric_name} failed to be used by FairnessModule"
                )
                failed_metrics.append(metric_name)

        return (performance_metrics, prob_metrics, failed_metrics)

    def _validate_arguments(self):
        check_existence(self.metrics, "metrics")
        check_data_instance(self.assessment_data, TabularData)
        check_artifact_for_nulls(self.assessment_data, "Data")
