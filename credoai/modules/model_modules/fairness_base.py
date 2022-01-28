from credoai.utils.metric_constants import (
    BINARY_CLASSIFICATION_METRICS, FAIRNESS_METRICS, 
    PROBABILITY_METRICS, METRIC_EQUIVALENTS
)
from credoai.utils.common import to_array, NotRunError
from credoai.utils.metric_utils import standardize_metric_name 
from credoai.modules.credo_module import CredoModule
from fairlearn.metrics import MetricFrame
from scipy.stats import norm
from sklearn.utils import check_consistent_length

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
        list of metric names as string or list of FairnessFunctions.
        Metric strings should in list returned by credoai.utils.list_metrics.
        Note for performance parity metrics like 
        "false negative rate parity" just list "false negative rate". Partiy metrics
        are calculated automatically.
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
        self.metric_conversions = None
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
        disaggregated_results = self.get_disaggregated_results()
        self.results = {'fairness': fairness_results,
                        'disaggregated_results': disaggregated_results}
        return self
        
    def prepare_results(self, method='between_groups', filter=None):
        """prepares fairness and disaggregated results to Credo AI

        Parameters
        ----------
        method : str, optional  
            How to compute the differences: "between_groups" or "to_overall". 
            See fairlearn.metrics.MetricFrame.difference
            for details, by default 'between_groups'
        filter : str, optional
            Regex string to filter fairness results if only a subset are desired.
            Passed as a regex argument to pandas `filter` function applied to the
            concatenated output of Fairnessmodule.get_fairness_results and
            Fairnessmodule.get_disaggregated_results, by default None

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
            disaggregated_df = self.results['disaggregated_results']
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
            list of metric names as string or list of FairnessFunctions.
            Metric strings should be included in ALL_METRICS
            found in credo.utils.metric_constants. Note for performance parity metrics like 
            "false negative rate parity" or "recall parity" just list "recall"
        """
        if replace:
            self.metrics = metrics
        else:
            self.metrics += metrics
        (self.performance_metrics,
         self.prob_metrics,
         self.fairness_metrics,
         self.fairness_prob_metrics,
         self.metric_conversions,
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
        for metric, func in self.fairness_metrics.items():
            results[metric] = func(y_true=self.y_true,
                                   y_pred=self.y_pred,
                                   sensitive_features=self.sensitive_features,
                                   method=method)
        for metric, func in self.fairness_prob_metrics.items():
            results[metric] = func(y_true=self.y_true,
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

    def get_disaggregated_results(self, melt=False):
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
        """Standardize and separates metrics

        Parameters
        ----------
        metrics : list-like. 

        Returns
        -------
        Separate dictionaries and lists of metrics
        """
        metric_conversions = {}
        custom_function_metrics = []
        for m in metrics:
            if type(m) == str:
                metric_conversions[m] = standardize_metric_name(m)
            else:
                custom_function_metrics.append(m)

        # separate metrics
        failed_metrics = []
        performance_metrics = {}
        prob_metrics = {}
        fairness_metrics = {}
        fairness_prob_metrics = {}
        for orig, standard in metric_conversions.items():
            if standard in PROBABILITY_METRICS:
                prob_metrics[orig] = BINARY_CLASSIFICATION_METRICS[standard]
            elif standard in BINARY_CLASSIFICATION_METRICS:
                performance_metrics[orig] = BINARY_CLASSIFICATION_METRICS[standard]
            elif standard in FAIRNESS_METRICS:
                fairness_metrics[orig] = FAIRNESS_METRICS[standard]
            else:
                failed_metrics.append(orig)

        # organize and standardize custom metrics
        for metric in custom_function_metrics:
            standard_name = standardize_metric_name(metric.name)
            metric_conversions[metric.name] = standard_name
            if metric.takes_sensitive_features:
                if metric.takes_prob:
                    fairness_prob_metrics[metric.name] = metric.func
                else:
                    fairness_metrics[metric.name] = metric.func
            else:
                if metric.takes_prob:
                    prob_metrics[metric.name] = metric.func
                else:
                    performance_metrics[metric.name] = metric.func
        return (performance_metrics, prob_metrics,
                fairness_metrics, fairness_prob_metrics,
                metric_conversions, failed_metrics)

    def _create_metric_frame(self, metrics, y_pred):
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
        
class FairnessFunction:
    def __init__(self, name, func, takes_sensitive_features=False, takes_prob=False):
        """A simple wrapper to define fairness functions

        A fairness function can have various signatures, which sould
        be reflected by the `takes_sensitive_features` and `takes_prob`
        arguments.

        Parameters
        ----------
        name : str
            The name of the function
        func : callable
            The function to use to calculate metrics. This function must be callable 
            as fn(y_true, y_pred / y_prob) or fn(y_true, y_pred, sensitive_features) 
            if `takes_sensitive_features` is True
        takes_sensitive_features : bool, optional
            Whether the function takes a sensitive_features parameter,
            as in fairlearn.metrics.equalized_odds_difference. Typically
            the function compares between groups in some way, by default False
        takes_prob : bool, optional
            Whether the function takes the decision probabilities
            vs. the predicted class, as for ROC AUC. by default False
        """
        self.name = name
        self.func = func
        self.takes_sensitive_features = takes_sensitive_features
        self.takes_prob = takes_prob
        
