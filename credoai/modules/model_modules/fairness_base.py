
from credoai.utils.common import to_array, NotRunError, ValidationError
from credoai.metrics import Metric, find_metrics, MODEL_METRIC_CATEGORIES 
from credoai.modules.credo_module import CredoModule
from fairlearn.metrics import MetricFrame
from scipy.stats import norm
from sklearn.utils import check_consistent_length
from typing import List, Union
import pandas as pd

class FairnessModule(CredoModule):
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
    sensitive_features :  (List, pandas.Series, numpy.ndarray)
        The sensitive features which should be used to create subgroups.
    y_true : (List, pandas.Series, numpy.ndarray)
        The ground-truth labels (for classification) or target values (for regression).
    y_pred : (List, pandas.Series, numpy.ndarray)
        The predicted labels for classification
    y_prob : (List, pandas.Series, numpy.ndarray), optional
        The unthresholded predictions, confidence values or probabilities.
    """

    def __init__(self,
                 metrics,
                 sensitive_features,
                 y_true,
                 y_pred,
                 y_prob=None
                 ):
        super().__init__()
        # data variables
        self.y_true = to_array(y_true)
        self.y_pred = to_array(y_pred)
        self.y_prob = to_array(y_prob) if y_prob is not None else None
        self.sensitive_features = sensitive_features
        self._validate_inputs()
        
        # assign variables
        self.metrics = metrics
        self.metric_frames = {}
        self.performance_metrics = None
        self.prob_metrics = None
        self.fairness_metrics = None
        self.fairness_prob_metrics = None
        self.failed_metrics = None
        self.update_metrics(metrics)

    def run(self, method='between_groups'):
        """
        Run fairness base module
        
        Parameters
        ----------
        method : str, optional
            How to compute the differences: "between_groups" or "to_overall". 
            See fairlearn.metrics.MetricFrame.difference
            for details, by default 'between_groups'
            
        Returns
        -------
        dict
            Dictionary containing two pandas Dataframes:
                - "disaggregated results": The disaggregated performance metrics, along with acceptability and risk
            as columns
                - "fairness": Dataframe with fairness metrics, along with acceptability and risk
            as columns
        """
        fairness_results = self.get_fairness_results(method=method)
        disaggregated_performance = self.get_disaggregated_performance()
        self.results = {'fairness': fairness_results,
                        'disaggregated_performance': disaggregated_performance}
        return self
        
    def prepare_results(self, filter=None):
        """prepares fairness and disaggregated results to Credo AI

        Structures results for export as a dataframe with appropriate structure
        for exporting. See credoai.modules.credo_module.

        Parameters
        ----------
        filter : str, optional
            Regex string to filter fairness results if only a subset are desired.
            Passed as a regex argument to pandas `filter` function applied to the
            concatenated output of Fairnessmodule.get_fairness_results and
            Fairnessmodule.get_disaggregated_performance, by default None

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        NotRunError
            Occurs if self.run is not called yet to generate the raw assessment results
        """
        if self.results is not None:
            # melt disaggregated df before combinding
            disaggregated_df = self.results['disaggregated_performance']
            disaggregated_df = disaggregated_df.reset_index()\
                .melt(id_vars=disaggregated_df.index.name, var_name='metric_type')\
                .set_index('metric_type')
            disaggregated_df['kind'] = 'disaggregated'
            # combine
            results = pd.concat([self.results['fairness'], disaggregated_df])
            if filter:
                results = results.filter(regex=filter)
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
        (self.performance_metrics,
         self.prob_metrics,
         self.fairness_metrics,
         self.fairness_prob_metrics,
         self.failed_metrics) = self._process_metrics(self.metrics)
        self._setup_metric_frames()

    def get_df(self):
        """Return dataframe of input arrays

        Returns
        -------
        pandas.DataFrame
            Dataframe containing the input arrays
        """
        df = pd.DataFrame({'sensitive': self.sensitive_features,
                           'true': self.y_true,
                           'pred': self.y_pred}).reset_index(drop=True)
        if self.y_prob is not None:
            y_prob_df = pd.DataFrame(self.y_prob)
            y_prob_df.columns = [f'y_prob_{i}' for i in range(y_prob_df.shape[1])]
            df = pd.concat([df, y_prob_df], axis=1)
        return df
    
    def get_overall_metrics(self):
        """Return performance metrics for each group

        Returns
        -------
        pandas.Series
            The overall performance metrics
        """
        return pd.concat([metric_frame.overall
                          for metric_frame in self.metric_frames.values()], axis=0)

    def get_fairness_results(self, method='between_groups'):
        """Return fairness and performance parity metrics

        Note, performance parity metrics are labeled with their
        related performance label, but are computed using 
        fairlearn.metrics.MetricFrame.difference(method)

        Parameters
        ----------
        method : str, optional
            How to compute the differences: "between_groups" or "to_overall".  
            See fairlearn.metrics.MetricFrame.difference
            for details, by default 'between_groups'

        Returns
        -------
        pandas.DataFrame
            The returned fairness metrics
        """

        results = {}
        for metric_name, metric in self.fairness_metrics.items():
            results[metric_name] = metric.fun(y_true=self.y_true,
                                              y_pred=self.y_pred,
                                              sensitive_features=self.sensitive_features,
                                              method=method)
        for metric_name, metric in self.fairness_prob_metrics.items():
            results[metric_name] = metric.fun(y_true=self.y_true,
                                              y_prob=self.y_prob,
                                              sensitive_features=self.sensitive_features,
                                              method=method)
        results = pd.Series(results, dtype=float, name='value')
        # add parity results
        parity_results = pd.Series(dtype=float)
        for metric_frame in self.metric_frames.values():
            parity_results = pd.concat(
                [parity_results, metric_frame.difference(method=method)])
        parity_results.name = 'value'

        results = pd.concat([results, parity_results]).convert_dtypes().to_frame()
        results.index.name = 'metric_type'
        # add kind
        results['kind'] = ['fairness'] * len(results)
        results.loc[results.index[-len(parity_results):], 'kind'] = 'parity'
        return results

    def get_disaggregated_performance(self, melt=False):
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
        disaggregated_df = pd.DataFrame()
        for metric_frame in self.metric_frames.values():
            df = metric_frame.by_group.copy().convert_dtypes()
            df.loc['overall', :] = metric_frame.overall
            if melt:
                df = df.reset_index()\
                    .melt(id_vars=df.index.name, var_name='metric_type')\
                    .set_index('metric_type')
                df['kind'] = 'disaggregated'
            disaggregated_df = pd.concat([disaggregated_df, df], axis=1)
        return disaggregated_df

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
                else:
                    raise Exception("Returned multiple metrics when searching using a metric name. Expected to find only one matching metric")
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
                failed_metrics.append(metric_name)

        return (performance_metrics, prob_metrics,
                fairness_metrics, fairness_prob_metrics,
                failed_metrics)

    def _create_metric_frame(self, metrics, y_pred):
        """Creates metric frame from dictionary of key:Metric"""
        metrics = {name: metric.fun for name, metric in metrics.items()}
        return MetricFrame(metrics=metrics,
                           y_true=self.y_true,
                           y_pred=y_pred,
                           sensitive_features=self.sensitive_features)
    
    def _setup_metric_frames(self):
        self.metric_frames = {}
        if self.y_pred is not None and self.performance_metrics:
            self.metric_frames['pred'] = self._create_metric_frame(
                self.performance_metrics, self.y_pred)
        # for metrics that require the probabilities
        self.prob_metric_frame = None
        if self.y_prob is not None and self.prob_metrics:
            self.metric_frames['prob'] = self._create_metric_frame(
                self.prob_metrics, self.y_prob)
            
    def _validate_inputs(self):
        check_consistent_length(self.y_true, self.y_pred,
                                self.y_prob, self.sensitive_features)
        

        
