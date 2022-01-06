import numpy as np
import scipy.stats as st
from fairlearn.metrics._disparities import _get_eo_frame
from fairlearn.metrics import make_derived_metric, true_positive_rate

from sklearn import metrics as sk_metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import check_consistent_length


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
    denominator = 1 + z ** 2 / n
    centre_adjusted_probability = p + z * z / (2 * n)
    adjusted_standard_deviation = np.sqrt(
        (p * (1 - p) + z * z / (4 * n))
    ) / np.sqrt(n)
    lower_bound = (
        centre_adjusted_probability - z * adjusted_standard_deviation
    ) / denominator
    upper_bound = (
        centre_adjusted_probability + z * adjusted_standard_deviation
    ) / denominator
    return np.array([lower_bound, upper_bound])


def wilson_ci(num_hits, num_total, confidence=0.95):
    """ Convenience wrapper for general_wilson """
    z = st.norm.ppf((1+confidence)/2)
    p = num_hits / num_total
    return general_wilson(p, num_total, z=z)


def confusion_wilson(y_true, y_pred, metric='tpr', confidence=0.95):
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
    if metric == 'true_positive_rate':
        numer = tp
        denom = positives
    elif metric == 'true_negative_rate':
        numer = tn
        denom = negatives
    elif metric == 'false_positive_rate':
        numer = fp
        denom = negatives
    elif metric == 'false_negative_rate':
        numer = fn
        denom = positives
    else:
        raise ValueError("""
        Metric must be one of the following:
            -true_positive_rate
            -true_negative_rate
            -false_positive_rate
            -false_negative_rate
        """)

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
    """Compute the false discovery rate.

    False omission rate is ``fn / (tn + fn)`` where ``fn`` is the number of
    false negatives and ``tn`` the number of true negatives. The false discovery rate is
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
        y_true,
        y_pred,
        *,
        sensitive_features,
        method='between_groups',
        sample_weight=None) -> float:
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
    fun = make_derived_metric(
        metric=true_positive_rate, transform='difference')
    return fun(y_true, y_pred, sensitive_features=sensitive_features, method=method, sample_weight=sample_weight)
