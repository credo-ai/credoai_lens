from credoai.utils.metric_constants import (
    BINARY_CLASSIFICATION_METRICS, FAIRNESS_METRICS, 
    PROBABILITY_METRICS, METRIC_EQUIVALENTS
)
from credoai.utils.metric_utils import standardize_metric_name 
from credoai.modules.credo_module import CredoModule
from fairlearn.metrics import MetricFrame
from scipy.stats import norm
from sklearn.utils import check_consistent_length

import pandas as pd

DEFAULT_BOUNDS = (float('-inf'), float('inf'))

class FairnessModule(CredoModule):
    """
    Fairness module for Credo AI. Handles any metric that can be
    calculated on a set of ground truth labels and predictions, 
    e.g., binary classification, multiclass classification, regression.

    This module takes in a set of metrics and "fairness bounds"
    defining acceptability limits and provides functionality to:
    - calculate the metrics
    - determine whether metrics are in bounds and calculate a "risk" score for them
    - create disaggregated metrics
    - create performance plots

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
    fairness_bounds : dict
        dictionary mapping fairness metric names to acceptability bounds
    performance_bounds : dict
        dictionary mapping performance metrics names to acceptability bounds.
        These bounds will be enforced on the disaggregated performance metrics.
        That is, the performance metric for each subgroup must be above the value
        to be deemed accceptable.

    Example
    ---------
    metrics = ['statistical parity']
    fairness_bounds = {'statistical parity': (-.1, .1)}
    module = Fairnessmodule(metrics, fairness_bounds, 'race', ...)
    """

    def __init__(self,
                 metrics,
                 sensitive_features,
                 y_true,
                 y_pred,
                 y_prob=None,
                 fairness_bounds=None,
                 performance_bounds=None
                 ):
        # data variables
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.sensitive_features = sensitive_features
        check_consistent_length(self.y_true, self.y_pred,
                                self.y_prob, self.sensitive_features)
        
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

        self.fairness_bounds = {standardize_metric_name(k): v
                                for k, v in (fairness_bounds or {}).items()}
        self.performance_bounds = {standardize_metric_name(k): v
                                   for k, v in (performance_bounds or {}).items()}

    def run(self, calculate_risk=False, method='between_groups', desired_mass=0.8):
        """
        Run fairness base module
        
        Parameters
        ----------
        calculate_risk : bool, optional
            Whether to include acceptability and risk in output,
            by default True
        method : str, optional
            How to compute the differences: "between_groups" or "to_overall". 
            See fairlearn.metrics.MetricFrame.difference
            for details, by default 'between_groups'
        desired_mass : float, optional
            Risk is modeled as a normal distribution such that
            "desired_mass" percent of the distribution's mass is 
            between the metrics thresholds. A higher number makes
            the thresholds more absolute. A lower number makes the 
            thresholds more permissive. By default 0.8
            
        Returns
        -------
        dict
            Dictionary containing two pandas Dataframes:
                - "disaggregated results": The disaggregated performance metrics, along with acceptability and risk
            as columns
                - "fairness": Dataframe with fairness metrics, along with acceptability and risk
            as columns
        """
        fairness_results = self.get_fairness_results(
            calculate_risk=calculate_risk,
            desired_mass=desired_mass,
            method=method)
        disaggregated_results = self.get_disaggregated_results(
            calculate_risk=calculate_risk,
            desired_mass=desired_mass)
        return {'fairness': fairness_results,
                'disaggregated_results': disaggregated_results}
        
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
            concatenated output of Fairnessmodule._get_fairness_metrics and
            Fairnessmodule._get_disaggregated_metrics, by default None

        Returns
        -------
        pd.DataFrame
        """
        fairness_results = self._get_fairness_metrics(method=method)
        disaggregated_results = self._get_disaggregated_metrics(melt=True)
        results = pd.concat([fairness_results, disaggregated_results])
        if filter:
            results = results.filter(regex=filter)
        return results
    
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

    def update_fairness_bounds(self, fairness_bounds):
        """replace metrics

        Parameters
        ----------
        fairness_bounds : dict
            dictionary mapping fairness metric names to acceptability bounds
        """
        self.fairness_bounds = {standardize_metric_name(k): v
                                for k, v in fairness_bounds.items()}

    def update_performance_bounds(self, performance_bounds):
        """replace metrics

        Parameters
        ----------
        performance_bounds : dict
            dictionary mapping performance metrics names to acceptability bounds.
            These bounds will be enforced on the disaggregated performance metrics.
            That is, the performance metric for each subgroup must be above the value
            to be deemed accceptable.
        """
        self.performance_bounds = {standardize_metric_name(k): v
                                   for k, v in performance_bounds.items()}

    def get_df(self):
        """Return dataframe of input arrays

        Returns
        -------
        pandas.DataFrame
            Dataframe containing the input arrays
        """
        df = pd.DataFrame({'sensitive': self.sensitive_features,
                           'true': self.y_true,
                           'pred': self.y_pred,
                           'prob': self.y_prob})
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

    def get_disaggregated_results(self, calculate_risk=True, desired_mass=0.8):
        """Return performance metrics for each group along with acceptability and risk

        Acceptability is defined by performance_bounds. A metric
        score is acceptable if it is within the bounds. Risk
        is a function of the thresholds and the "desired mass" parameter.

        Parameters
        ----------
        calculate_risk : bool, optional
            Whether to include acceptability and risk in output,
            by default True
        desired_mass : float, optional
            Risk is modeled as a normal distribution such that
            "desired_mass" percent of the distribution's mass is 
            between the metrics thresholds. A higher number makes
            the thresholds more absolute. A lower number makes the 
            thresholds more permissive. By default 0.8

        Returns
        -------
        pandas.DataFrame
            The disaggregated performance metrics, along with acceptability and risk
            as columns
        """
        disaggregated_results = self._get_disaggregated_metrics()
        if calculate_risk:
            for name, col in disaggregated_results.iteritems():
                risks = []
                for _, value in col.items():
                    risks.append(self._calc_risk(
                        name, value, self.performance_bounds, desired_mass=desired_mass))
                disaggregated_results[[
                    f'{name}_risk',
                    f'{name}_acceptable',
                    f'{name}_bound_lower',
                    f'{name}_bound_upper']] = risks
        return disaggregated_results.sort_index(axis=1)

    def get_fairness_results(self, calculate_risk=True, method='between_groups', desired_mass=0.8):
        """Return fairness metrics with acceptability and risk

        Acceptability is defined by fairness_bounds. A metric
        score is acceptable if it is within the bounds. Risk
        is a function of the thresholds and the "desired mass" parameter.


        Parameters
        ----------
        calculate_risk : bool, optional
            Whether to include acceptability and risk in output,
            by default True
        method : str, optional
            How to compute the differences: "between_groups" or "to_overall".  
            See fairlearn.metrics.MetricFrame.difference
            for details, by default 'between_groups'
        desired_mass : float, optional
            Risk is modeled as a normal distribution such that
            "desired_mass" percent of the distribution's mass is 
            between the metrics thresholds. A higher number makes
            the thresholds more absolute. A lower number makes the 
            thresholds more permissive. By default 0.8

        Returns
        -------
        pandas.DataFrame
            Dataframe with fairness metrics, along with acceptability and risk
            as columns
        """
        fairness_results = self._get_fairness_metrics(method).to_frame('value')
        if calculate_risk:
            risks = []
            for name, value in fairness_results['value'].items():
                risks.append(self._calc_risk(
                    name, value, self.fairness_bounds, desired_mass=desired_mass))
            fairness_results[['risk', 'acceptable',
                              'bound_lower', 'bound_upper']] = risks
        return fairness_results.sort_index()

    def get_unacceptable_metrics(self, method='between_groups'):
        """Return unacceptable disaggregated metrics

        If performance_bounds are set, return disaggregated metrics that fail. This
        will return particular group/metric combinations that fail.

        If performance_bounds are not set, return performance metrics
        that showed unacceptable performance parity metrics (defined by fairness_bounds). 
        In this case metrics for every group will be shown.

        Parameters
        ----------
        method : str, optional
            How to compute the differences: "between_groups" or "to_overall". 
            See fairlearn.metrics.MetricFrame.difference
            for details, by default 'between_groups'

        Returns
        -------
        pandas.DataFrame
            Unacceptable performance metrics disaggregated across groups
        """
        if self.performance_bounds:
            disaggregated_results = self.get_disaggregated_results()
            unacceptable_results = ~disaggregated_results.filter(
                regex='acceptable')
            unacceptable_metrics = unacceptable_results.any()
            unacceptable_metrics_regex = '|'.join([i.replace('_acceptable', '')
                                                   for i in unacceptable_metrics[unacceptable_metrics].index])
            unacceptable_groups = unacceptable_results.any(axis=1)
            if unacceptable_metrics_regex:
                return disaggregated_results[unacceptable_groups].filter(regex=unacceptable_metrics_regex, axis=1)
            else:
                return pd.DataFrame()
        else:
            disaggregated_df = self._get_disaggregated_metrics()
            disaggregated_metrics = disaggregated_df.columns
            unacceptable_metrics = self.get_fairness_results(method=method) \
                .query('acceptable == False')
            tmp = unacceptable_metrics.index.intersection(
                disaggregated_metrics)
            return disaggregated_df.loc[:, tmp]

    def prepare_fairness_results(self, method='between_groups', filter=None):
        """prepares fairness results to Credo AI

        Parameters
        ----------
        method : str, optional  
            How to compute the differences: "between_groups" or "to_overall".  
            See fairlearn.metrics.MetricFrame.difference
            for details, by default 'between_groups'
        filter : str, optional
            Regex string to filter fairness results if only a subset are desired.
            Passed as a regex argument to pandas `filter` function applied to the
            output of Fairnessmodule._get_fairness_metrics, by default None

        Returns
        -------
        pd.DataFrame
        """
        fairness_results = self._get_fairness_metrics(method=method)
        if filter:
            results = results.filter(regex=filter)
        return results

    def prepare_disaggregated_results(self, filter=None):
        """prepares disaggregated results to Credo AI

        Parameters
        ----------
        filter : str, optional
            Regex string to filter disaggregated results if only a subset are desired.
            Passed as a regex argument to pandas `filter` function applied to the
            output of Fairnessmodule._get_disaggregated_metrics, by default None

        Returns
        -------
        pd.DataFrame
        """
        disaggregated_results = self._get_disaggregated_metrics(melt=True)
        if filter:
            results = results.filter(regex=filter)
        return results

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

    def _calc_risk(self, name, value, bounds_dict, desired_mass=0.8):
        name = self.metric_conversions.get(name, name)
        bounds = bounds_dict.get(name, DEFAULT_BOUNDS)
        within_bounds = bounds[0] <= value <= bounds[1]
        _range = (bounds[1]-bounds[0])
        center = bounds[0] + _range/2
        # this standard deviation ensures that
        # desired_mass% of the distribution's mass
        # is between the two bounds
        # e.g., higher desired_mass leads to stronger
        # thresholds
        desired_mass = min(desired_mass, .99)
        scaling = (1/abs(norm.ppf((1-desired_mass)/2)))
        sd = _range/2*scaling
        dist = norm(center, sd)

        # transform so that positive and negative
        # deviations from center act identically
        value_transform = center + abs(center-value)
        risk = (dist.cdf(value_transform)-.5)*2
        return risk, within_bounds, bounds[0], bounds[1]

    def _get_fairness_metrics(self, method='between_groups'):
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
        results = pd.Series(results, dtype=float)
        # add parity results
        parity_results = pd.Series(dtype=float)
        for metric_frame in self.metric_frames.values():
            parity_results = pd.concat(
                [parity_results, metric_frame.difference(method=method)])
        results = pd.concat([results, parity_results]).convert_dtypes()
        results.name = 'metric'
        return results

    def _get_disaggregated_metrics(self, melt=False):
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
        if melt:
            disaggregated_df = pd.Series(dtype=float)
        else:
            disaggregated_df = pd.DataFrame()
        for metric_frame in self.metric_frames.values():
            df = metric_frame.by_group.copy().convert_dtypes()
            df.loc['overall', :] = metric_frame.overall
            if melt:
                melted_df = df.reset_index().melt(id_vars=df.index.name)
                # create index
                index = melted_df.loc[:, ['variable', df.index.name]] \
                    .astype(str) \
                    .agg(f'_{df.index.name}-'.join, axis=1)
                # create series
                df = pd.Series(melted_df['value'].tolist(), index=index)
                disaggregated_df = pd.concat([disaggregated_df, df])
            else:
                disaggregated_df = pd.concat([disaggregated_df, df], axis=1)
        return disaggregated_df
    
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