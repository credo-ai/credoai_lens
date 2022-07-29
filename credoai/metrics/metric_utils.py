import textwrap
from collections import defaultdict

import numpy as np
from credoai.metrics.metrics import (
    ALL_METRICS,
    METRIC_CATEGORIES,
    METRIC_NAMES,
    MODEL_METRIC_CATEGORIES,
)
from scipy.stats import norm
from sklearn.utils import resample


def list_metrics(verbose=True):
    metrics = defaultdict(set)
    for metric in ALL_METRICS:
        if metric.metric_category in MODEL_METRIC_CATEGORIES:
            metrics[metric.metric_category] |= metric.equivalent_names
    if verbose:
        for key, val in metrics.items():
            metric_str = textwrap.fill(
                ", ".join(sorted(list(val))),
                width=50,
                initial_indent="\t",
                subsequent_indent="\t",
            )
            print(key)
            print(metric_str)
            print("")
    return metrics


def bootstrap_CI(
    metric_fun,
    fun_inputs,
    CI=0.95,
    reps=1000,
    method="se",
    random_state=None,
    **fun_kwargs
):
    """Calculate boostrap CI for a metric

    Uses bootstrap resampling of input data to

    Parameters
    ----------
    metric_fun : callable
        Callable metric
    fun_inputs : dict
        Dictionary of function inputs that will be boostrapped over.
        E.g., {y_true: [...], y_pred: [...]}. Each value in the dictionary
        should be the same length
    CI : float, optional
        The confidence interval. Defaults to .95
    reps : int, optional
        Number of bootstrap samples to create, by default 1000
    method : str, optional
        method of calculate bootstrapped confidence interval:
            "se": standard error. This method uses the estimated mean
                  and standard error of the metric to calculate CI
            "percentile": This method takes the empirical quantiles directly
                          from the bootstrap distribution
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for shuffling
        the data.
    **fun_kwargs : kwargs
        set of key words that will be passed to metric_fun.
    """
    CI_bounds = [(1 - CI) / 2, 1 - (1 - CI) / 2]
    keys = fun_inputs.keys()
    data = list(fun_inputs.values())
    vals = []

    # perform bootstrap
    for _ in range(reps):
        sample = resample(*data)
        inputs = {k: v for k, v in zip(keys, sample)}
        vals.append(metric_fun(**inputs, **fun_kwargs))

    # two methods of calculating bootstrap CI
    if method == "percentile":
        CI_out = [np.percentile(vals, i * 100) for i in CI_bounds]
    elif method == "se":
        mu = np.mean(vals)
        se = np.std(vals)
        delta = norm.ppf(CI_bounds[1]) * se
        CI_out = [mu - delta, mu + delta]
    return CI_out
