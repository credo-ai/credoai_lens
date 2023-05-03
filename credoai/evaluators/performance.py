import numpy as np
import pandas as pd
from connect.evidence import MetricContainer, TableContainer
from sklearn.metrics import confusion_matrix

from credoai.artifacts import ClassificationModel
from credoai.evaluators.evaluator import Evaluator
from credoai.evaluators.utils.fairlearn import setup_metric_frames
from credoai.evaluators.utils.validation import check_data_for_nulls, check_existence
from credoai.modules.metrics import process_metrics
from credoai.utils.common import ValidationError


class Performance(Evaluator):
    """
    Performance evaluator for Credo AI.

    This evaluator calculates overall performance metrics.
    Handles any metric that can be calculated on a set of ground truth labels and predictions,
    e.g., binary classification, multi class classification, regression.

    This module takes in a set of metrics and provides functionality to:

    - calculate the metrics
    - create disaggregated metrics

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
    metrics : List-like
        list of metric names as strings or list of Metric objects (credoai.modules.metrics.Metric).
        Metric strings should in list returned by credoai.modules.metric_utils.list_metrics().
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

    required_artifacts = {"model", "assessment_data"}

    def __init__(self, metrics=None):
        super().__init__()
        # assign variables
        self.metrics = metrics
        self.metric_frames = {}
        self.processed_metrics = None

    def _validate_arguments(self):
        check_existence(self.metrics, "metrics")
        check_existence(self.assessment_data.y, "y")
        check_data_for_nulls(
            self.assessment_data, "Data", check_X=True, check_y=True, check_sens=False
        )

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
        """
        results = []
        overall_metrics = self.get_overall_metrics()
        threshold_metrics = self.get_overall_threshold_metrics()

        if overall_metrics is not None:
            results.append(overall_metrics)
        if threshold_metrics is not None:
            results += threshold_metrics

        if isinstance(self.model, ClassificationModel):
            results.append(self._create_confusion_container())
        self.results = results
        return self

    def update_metrics(self, metrics, replace=True):
        """replace metrics

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

        self.processed_metrics, _ = process_metrics(self.metrics, self.model.type)

        dummy_sensitive = pd.Series(["NA"] * len(self.y_true), name="NA")
        self.metric_frames = setup_metric_frames(
            self.processed_metrics,
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
        """Return scalar performance metrics for each group

        Returns
        -------
        pandas.Series
            The overall performance metrics
        """
        # retrieve overall metrics for one of the sensitive features only as they are the same
        overall_metrics = [
            metric_frame.overall
            for name, metric_frame in self.metric_frames.items()
            if name != "thresh"
        ]
        if not overall_metrics:
            return

        output_series = (
            pd.concat(overall_metrics, axis=0).rename(index="value").to_frame()
        )
        output_series = output_series.reset_index().rename({"index": "type"}, axis=1)

        return MetricContainer(output_series, **self.get_info())

    def get_overall_threshold_metrics(self):
        """Return performance metrics for each group

        Returns
        -------
        pandas.Series
            The overall performance metrics
        """
        # retrieve overall metrics for one of the sensitive features only as they are the same
        if not "thresh" in self.metric_frames:
            return

        threshold_results = (
            pd.concat([self.metric_frames["thresh"].overall], axis=0)
            .rename(index="value")
            .to_frame()
        )
        threshold_results = threshold_results.reset_index().rename(
            {"index": "threshold_metric"}, axis=1
        )
        threshold_results.name = "threshold_metric_performance"

        results = []
        for _, threshold_metric in threshold_results.iterrows():
            metric = threshold_metric.threshold_metric
            threshold_metric.value.name = "threshold_dependent_performance"
            results.append(
                TableContainer(
                    threshold_metric.value,
                    **self.get_info({"metric_type": metric}),
                )
            )

        return results

    def _create_confusion_container(self):
        confusion_container = TableContainer(
            create_confusion_matrix(self.y_true, self.y_pred),
            **self.get_info(),
        )
        return confusion_container


############################################
## Evaluation helper functions

## Helper functions create evidences
## to be passed to .evaluate to be wrapped
## by evidence containers
############################################
def create_confusion_matrix(y_true, y_pred):
    """Create a confusion matrix as a dataframe

    Parameters
    ----------
    y_true : pd.Series of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    """
    labels = np.unique(y_true)
    confusion = confusion_matrix(y_true, y_pred, normalize="true", labels=labels)
    confusion_df = pd.DataFrame(confusion, index=labels.copy(), columns=labels)
    confusion_df.index.name = "true_label"
    confusion_df = confusion_df.reset_index().melt(
        id_vars=["true_label"], var_name="predicted_label"
    )
    confusion_df.name = "Confusion Matrix"
    return confusion_df
