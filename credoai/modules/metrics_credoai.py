"""Custom metrics defined by Credo AI"""

from functools import partial

import numpy as np
import pandas as pd
import scipy.stats as st
from fairlearn.metrics import make_derived_metric, true_positive_rate
from sklearn import metrics as sk_metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import check_consistent_length
from typing_extensions import Literal


def multiclass_confusion_metrics(y_true, y_pred, metric=None, average="weighted"):
    """Calculate

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    metric : str, optional
        If provided, returns a specific metric. All metrics are returned if None is provided.
        Options are:
            "TPR": Sensitivity, hit rate, recall, or true positive rate
            "TNR": Specificity or true negative rate
            "PPV": Precision or positive predictive value
            "NPV": Negative predictive value
            "FPR": Fall out or false positive rate
            "FNR": False negative rate
            "FDR": False discovery rate
            "ACC": Overall accuracy
    average : str
        Options are "weighted", "macro" or None (which will return the values for each label)

    Returns
    -------
    dict or float
        dict if metric is not provided
    """
    cnf_matrix = confusion_matrix(y_true, y_pred)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    metrics = {
        "TPR": TP / (TP + FN),
        "TNR": TN / (TN + FP),
        "PPV": TP / (TP + FP),
        "NPV": TN / (TN + FN),
        "FPR": FP / (FP + TN),
        "FNR": FN / (TP + FN),
        "FDR": FP / (TP + FP),
        "ACC": (TP + TN) / (TP + FP + FN + TN),
    }
    if average == "weighted":
        weights = np.unique(y_true, return_counts=True)[1] / len(y_true)
        metrics = {k: np.average(v, weights=weights) for k, v in metrics.items()}
    elif average == "macro":
        metrics = {k: v.mean() for k, v in metrics.items()}
    if metric:
        return metrics[metric]
    else:
        return metrics


def general_wilson(p, n, z=1.96):
    """Return lower and upper bound using Wilson Interval.
    Parameters
    ----------
    p : float
        Proportion of successes.
    n : int
        Total number of trials.
    digits : int
        Digits of precisions to which the returned bound will be rounded
    z : float
        Z-score, which indicates the number of standard deviations of confidence
    Returns
    -------
        np.ndarray
        Array of length 2 of form: [lower_bound, upper_bound]
    """
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z * z / (2 * n)
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z * z / (4 * n))) / np.sqrt(n)
    lower_bound = (
        centre_adjusted_probability - z * adjusted_standard_deviation
    ) / denominator
    upper_bound = (
        centre_adjusted_probability + z * adjusted_standard_deviation
    ) / denominator
    return np.array([lower_bound, upper_bound])


def wilson_ci(num_hits, num_total, confidence=0.95):
    """Convenience wrapper for general_wilson"""
    z = st.norm.ppf((1 + confidence) / 2)
    p = num_hits / num_total
    return general_wilson(p, num_total, z=z)


def confusion_wilson(y_true, y_pred, metric="tpr", confidence=0.95):
    """Return Wilson Interval bounds for performance metrics

    Metrics derived from confusion matrix

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels
    y_pred : array-like of shape (n_samples,)
        Predicted labels
    metric : string
        indicates kind of performance metric. Must be
        tpr, tnr, fpr, or fnr. "tpr" is true-positive-rate,
        "fnr" is false negative rate, etc.
    Returns
    -------
        np.ndarray
        Array of length 2 of form: [lower_bound, upper_bound]
    """
    check_consistent_length(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    negatives = tn + fp
    positives = tp + fn
    if metric == "true_positive_rate":
        numer = tp
        denom = positives
    elif metric == "true_negative_rate":
        numer = tn
        denom = negatives
    elif metric == "false_positive_rate":
        numer = fp
        denom = negatives
    elif metric == "false_negative_rate":
        numer = fn
        denom = positives
    else:
        raise ValueError(
            """
        Metric must be one of the following:
            -true_positive_rate
            -true_negative_rate
            -false_positive_rate
            -false_negative_rate
        """
        )

    bounds = wilson_ci(numer, denom, confidence)
    return bounds


def accuracy_wilson(y_true, y_pred, confidence=0.95):
    """Return Wilson Interval bounds for accuracy metric.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels
    y_pred : array-like of shape (n_samples,)
        Predicted labels
    Returns
    -------
        np.ndarray
        Array of length 2 of form: [lower_bound, upper_bound]
    """
    check_consistent_length(y_true, y_pred)
    score = accuracy_score(y_true, y_pred)
    bounds = general_wilson(score, len(y_true), confidence)
    return bounds


# metric definitions


def false_discovery_rate(y_true, y_pred, **kwargs):
    """Compute the false discovery rate.

    False discovery rate is 1-precision, or ``fp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The false discovery rate is
    intuitively the rate at which the classifier will be wrong when
    labeling an example as positive.

    The best value is 0 and the worst value is 1.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    kwargs :  key, value mappings
        Other keyword arguments are passed through
        to scikit-learn.metrics.precision
    """
    return 1.0 - sk_metrics.precision_score(y_true, y_pred, **kwargs)


def false_omission_rate(y_true, y_pred, **kwargs):
    """Compute the false omission rate.

    False omission rate is ``fn / (tn + fn)`` where ``fn`` is the number of
    false negatives and ``tn`` the number of true negatives. The false omission rate is
    intuitively the rate at which the classifier will be wrong when
    labeling an example as negative.

    The best value is 0 and the worst value is 1.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    kwargs :  key, value mappings
        Other keyword arguments are passed through
        to scikit-learn.metrics.precision
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tn)


def equal_opportunity_difference(
    y_true, y_pred, *, sensitive_features, method="between_groups", sample_weight=None
) -> float:
    """Calculate the equal opportunity difference.

    Equivalent to the `true_positive_rate_difference` defined as the difference between the
    largest and smallest of :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a`
    of the sensitive feature(s).

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.
    sensitive_features :
        The sensitive features over which demographic parity should be assessed
    method : str
        How to compute the differences. See :func:`fairlearn.metrics.MetricFrame.ratio`
        for details.
    sample_weight : array-like
        The sample weights
    Returns
    -------
    float
        The average odds difference
    """
    fun = make_derived_metric(metric=true_positive_rate, transform="difference")
    return fun(
        y_true,
        y_pred,
        sensitive_features=sensitive_features,
        method=method,
        sample_weight=sample_weight,
    )


def ks_statistic(y_true, y_pred) -> float:
    """Performs the two-sample Kolmogorov-Smirnov test (two-sided) for a regression model.

    The test compares the underlying continuous distributions F(x) and G(x) of two independent samples.
    The null hypothesis is that the two distributions are identical, F(x)=G(x)
    If the KS statistic is small or the p-value is high,
    then we cannot reject the null hypothesis in favor of the alternative.

    For practical purposes, if the statistic value is higher than the critical value, the two distributions are different.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    Returns
    -------
    float
        KS statistic value
    """

    ks_stat = st.ks_2samp(y_true, y_pred).statistic

    return ks_stat


def ks_statistic_binary(y_true, y_pred) -> float:
    """Performs the two-sample Kolmogorov-Smirnov test (two-sided) for binary classifiers.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted probabilities returned by the classifier.

    Returns
    -------
    float
        KS statistic value
    """

    df = pd.DataFrame({"real": y_true, "proba": y_pred})
    # Recover each class
    class0 = df[df["real"] == 0]
    class1 = df[df["real"] == 1]
    ks_stat = st.ks_2samp(class0["proba"], class1["proba"]).statistic

    return ks_stat


def interpolate_increasing_thresholds(lib_thresholds, *series):
    out = [list() for i in series]
    quantization = 1 / (
        len(lib_thresholds) * (max(lib_thresholds) - min(lib_thresholds))
    )
    interpolated_thresholds = np.arange(
        min(lib_thresholds), max(lib_thresholds), quantization
    )

    for t in interpolated_thresholds:
        if t >= lib_thresholds[0]:
            lib_thresholds.pop(0)
            for s in series:
                s.pop(0)
        for i, s in enumerate(out):
            s.append(series[i][0])

    return out + [interpolated_thresholds]


def interpolate_decreasing_thresholds(lib_thresholds, *series):
    out = [list() for i in series]
    quantization = -1 / (
        len(lib_thresholds) * (max(lib_thresholds) - min(lib_thresholds))
    )
    interpolated_thresholds = np.arange(
        max(lib_thresholds), min(lib_thresholds), quantization
    )

    for t in interpolated_thresholds:
        for i, s in enumerate(out):
            s.append(series[i][0])
        if t <= lib_thresholds[0]:
            lib_thresholds.pop(0)
            for s in series:
                s.pop(0)

    return out + [interpolated_thresholds]


def credo_pr_curve(y_true, y_prob):
    p, r, t = sk_metrics.precision_recall_curve(y_true, y_prob)
    (
        precision,
        recall,
        thresholds,
    ) = interpolate_increasing_thresholds(t.tolist(), p.tolist(), r.tolist())
    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": thresholds,
        }
    )


def credo_roc_curve(y_true, y_prob):
    fpr, tpr, thresh = sk_metrics.roc_curve(y_true, y_prob)
    (
        false_positive_rate,
        true_positive_rate,
        thresholds,
    ) = interpolate_decreasing_thresholds(thresh.tolist(), fpr.tolist(), tpr.tolist())
    return pd.DataFrame(
        {
            "false_positive_rate": false_positive_rate,
            "true_positive_rate": true_positive_rate,
            "threshold": thresholds,
        }
    )


def credo_det_curve(y_true, y_prob):
    fpr, fnr, t = sk_metrics.det_curve(y_true, y_prob)
    (
        false_positive_rate,
        false_negative_rate,
        thresholds,
    ) = interpolate_increasing_thresholds(t.tolist(), fpr.tolist(), fnr.tolist())
    return pd.DataFrame(
        {
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "thresholds": thresholds,
        }
    )


def gini_coefficient_discriminatory(y_true, y_prob, **kwargs):
    """Returns the Gini Coefficient of a discriminatory model

    NOTE: There are two popular, yet distinct metrics known as the 'gini coefficient'.

    The value calculated by this function provides a summary statistic for the Cumulative Accuracy Profile (CAP) curve.
    This notion of Gini coefficient (or Gini index) is a _discriminatory_ metric. It helps characterize the ordinal
    relationship between predictions made by a model and the ground truth values for each sample.

    This metric has a linear relationship with the area under the receiver operating characteristic curve:
        :math:`G = 2*AUC - 1`

    See https://towardsdatascience.com/using-the-gini-coefficient-to-evaluate-the-performance-of-credit-score-models-59fe13ef420
    for more details.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_prob : array-like
        Predicted probabilities returned by a call to the model's `predict_proba()` function.

    Returns
    -------
    float
        Discriminatory Gini Coefficient
    """
    G = (2 * sk_metrics.roc_auc_score(y_true, y_prob, **kwargs)) - 1
    return G


def population_stability_index(
    expected_array,
    actual_array,
    percentage=False,
    buckets: int = 10,
    buckettype: Literal["bins", "quantiles"] = "bins",
):
    """Calculate the PSI for a single variable.

    PSI is a measure of how much a distribution has changed over time or between
    two different samples of a population.
    It does this by bucketing the two distributions and comparing the percents of
    items in each of the buckets. The final result is a single number:

        :math:`PSI = \sum \left ( Actual_{%} - Expected_{%} \right ) \cdot ln\left ( \frac{Actual_{%}}{Expected_{%}} \right )`

    The common interpretations of the PSI result are:

    PSI < 0.1: no significant population change
    PSI < 0.25: moderate population change
    PSI >= 0.25: significant population change

    The number of buckets chosen and the bucketing logic affect the final result.


    References
    ----------
    Based on the code in: github.com/mwburke by Matthew Burke.
    For implementation walk through: https://mwburke.github.io/data%20science/2018/04/29/population-stability-index.html
    For a more theoretical reference: https://scholarworks.wmich.edu/cgi/viewcontent.cgi?article=4249&context=dissertations


    Parameters
    ----------
    expected_array: array-like
        Array of expected/initial values
    actual_array: array-like
        Array of new values
    percentage: bool
        When True the arrays are interpreted as already binned/aggregated. This is
        so that the user can perform their own aggregation and pass it directly to
        the metric. Default = False
    buckets: int
        number of percentile ranges to bucket the values into
    buckettype: Literal["bins", "quantiles"]
        type of strategy for creating buckets, bins splits into even splits,
        quantiles splits into quantile buckets

    Returns:
        psi_value: calculated PSI value
    """
    epsilon: float = 0.001

    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    if not percentage:
        # Define histogram breakpoints
        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == "bins":
            breakpoints = scale_range(
                breakpoints, np.min(expected_array), np.max(expected_array)
            )
        elif buckettype == "quantiles":
            breakpoints = np.stack(
                [np.percentile(expected_array, b) for b in breakpoints]
            )

        # Populate bins and calculate percentages
        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(
            expected_array
        )
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)
    else:
        expected_percents = expected_array
        actual_percents = actual_array

    # Substitute 0 with an arbitrary epsilon << 1
    # This is to avoid inf in the following calculations
    expected_percents[expected_percents == 0] = epsilon
    actual_percents[actual_percents == 0] = epsilon

    psi_values = (expected_percents - actual_percents) * np.log(
        expected_percents / actual_percents
    )

    return np.sum(psi_values)


def kl_divergence(p, q):
    """Calculates KL divergence of two discrete distributions

    Parameters
    ----------
    p : list
        first distribution
    q : list
        second distribution

    Returns
    -------
    float
        KL divergence

    Examples
    --------
    >>> p = [.05, .1, .2, .05, .15, .25, .08, .12]
    >>> q = [.30, .1, .2, .10, .10, .02, .08, .10]
    >>> round(kl_divergence(p, q),2)
    0.59
    """
    if len(p) != len(q):
        raise ValueError("`p` and `q` must have the same length.")

    EPSILON = 1e-12
    vals = []
    for pi, qi in zip(p, q):
        vals.append(pi * np.log((pi + EPSILON) / (qi + EPSILON)))

    return sum(vals)


def normalized_discounted_cumulative_kl_divergence(ranked_list, desired_proportions):
    """Calculates normalized discounted cumulative KL divergence (NDKL)

    It is non-negative, with larger values indicating a greater divergence between the
        desired and actual distributions of groups labels. It ranges  from 0 to inf
        and the ideal value is 0 for fairness.

    Parameters
    ----------
    ranked_list : list
        Ranked list of items
    desired_proportions : dict
        The desired proportion for each group

    Returns
    -------
    float
        normalized discounted cumulative KL divergence

    References
    ----------
    Geyik, Sahin Cem, Stuart Ambler, and Krishnaram Kenthapadi. "Fairness-aware ranking in search &
        recommendation systems with application to linkedin talent search."
        Proceedings of the 25th acm sigkdd international conference on knowledge discovery &
        data mining. 2019.

    Examples
    --------
    >>> ranked_list = ['female', 'male', 'male', 'female', 'male', 'male']
    >>> desired_proportions = {'female': 0.6, 'male': 0.4}
    >>> round(normalized_discounted_cumulative_kl_divergence(ranked_list, desired_proportions), 15)
    0.208096993149323
    """
    num_items = len(ranked_list)
    Z = np.sum(1 / (np.log2(np.arange(1, num_items + 1) + 1)))

    total = 0.0
    for k in range(1, num_items + 1):
        item_attr_k = list(ranked_list[:k])
        item_distr = [
            item_attr_k.count(attr) / len(item_attr_k)
            for attr in desired_proportions.keys()
        ]
        total += (1 / np.log2(k + 1)) * kl_divergence(
            item_distr, list(desired_proportions.values())
        )

    ndkl = (1 / Z) * total

    return ndkl


def skew_parity(items_list, desired_proportions, parity_type="difference"):
    """Calculates skew parity

    max_skew vs min_skew, where skew is the proportion of the selected
      items from a group over the desired proportion for that group.

    Parameters
    ----------
    ranked_list : list
        Ranked list of items
    desired_proportions : dict
        The desired proportion for each group
    parity_type : type pf parity to use. Two options:
        'difference' : `max_skew - min_skew`. It ranges from 0 to inf and the ideal value is 0.
        'ratio' : `min_skew / max_skew`. It ranges from 0 to 1 and the ideal value is 1.

    Returns
    -------
    float
        skew parity difference

    Examples
    --------
    >>> ranked_list = ['female', 'male', 'male', 'female', 'male', 'male']
    >>> desired_proportions = {'female': 0.6, 'male': 0.4}
    >>> skew_parity(ranked_list, desired_proportions)
    1.1111111111087035
    """
    EPSILON = 1e-12
    groups, counts = np.unique(items_list, return_counts=True)
    subset_proportions = dict(zip(groups, counts / len(items_list)))

    skew = {}
    for g in groups:
        sk = (subset_proportions[g] + EPSILON) / (desired_proportions[g] + EPSILON)
        skew[g] = sk

    if parity_type == "difference":
        parity = max(skew.values()) - min(skew.values())
    elif parity_type == "ratio":
        parity = min(skew.values()) / max(skew.values())
    else:
        raise ValueError(
            f"Possible values of `parity_type` are 'difference' and 'ratio'. You provided {parity_type}"
        )

    return parity


def credo_gain_chart(y_true, y_prob, bins=10):
    sort_indices = np.argsort(y_prob)[::-1]  # descending order
    total_events = np.sum(y_true)
    overall_rate = np.mean(y_true)

    sorted_y_true = y_true[sort_indices]

    chunked_y_true = np.array_split(sorted_y_true, bins)

    cumulative_samples = np.zeros(bins)
    cumulative_events = np.zeros(bins)
    cumulative_ratio = np.zeros(bins)
    lift = np.zeros(bins)

    for i in range(bins):
        cumulative_samples[i] = len(np.hstack(chunked_y_true[: i + 1]))
        cumulative_events[i] = np.sum(np.hstack(chunked_y_true[: i + 1]))
        cumulative_ratio[i] = cumulative_events[i] / total_events
        lift[i] = (cumulative_events[i] / cumulative_samples[i]) / overall_rate
    return pd.DataFrame(
        {
            "Decile": np.arange(bins) + 1,
            "Cumulative Samples": cumulative_samples,
            "Cumulative Events": cumulative_events,
            "Cumulative Gain": cumulative_ratio,
            "Lift": lift,
        }
    )
