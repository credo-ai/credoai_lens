
Feature Drift
=============


Measure Feature Drift using population stability index.

This evaluator measures feature drift in:

1. Model prediction: the prediction for the assessment dataset is compared
   to the prediction for the training dataset.
   In the case of classifiers, the prediction is performed with predict proba if available.
   If it is not available, the prediction is treated like a categorical variable, see the
   processing of categorical variables in the item below.

2. Dataset features: 1 to 1 comparison across all features for the datasets. This is also
   referred to as "characteristic stability index" (CSI). Features are processed depending
   on their type:

   - Numerical features are directly fed into the population_stability_index metric, and
     binned according to the parameters specified at init time.
   - Categorical features percentage distribution is manually calculated. The % amount of
     samples per each class is calculated and then fed into the population_stability_index metric.
     The percentage flag in the metric is set to True, to bypass the internal binning process.


Parameters
----------
buckets : int, optional
    Number of buckets to consider to bin the predictions, by default 10
buckettype :  Literal["bins", "quantiles"]
    Type of strategy for creating buckets, bins splits into even splits,
    quantiles splits into quantiles buckets, by default "bins"
csi_calculation : bool, optional
    Calculate characteristic stability index, i.e., PSI for all features in the datasets,
    by default False
