
Data fairness
=============


Data Fairness for Credo AI.

This evaluator performs a fairness evaluation on the dataset. Given a sensitive feature,
it calculates a number of assessments:

- group differences of features
- evaluates whether features in the dataset are proxies for the sensitive feature
- whether the entire dataset can be seen as a proxy for the sensitive feature
  (i.e., the sensitive feature is "redundantly encoded")

Parameters
----------
X : pandas.DataFrame
    The features
y : pandas.Series
    The outcome labels
sensitive_features : pandas.Series
    A series of the sensitive feature labels (e.g., "male", "female") which should be used to create subgroups
categorical_features_keys : list[str], optional
    Names of the categorical features
categorical_threshold : float
    Parameter for automatically identifying categorical columns. See
    `credoai.utils.common.is_categorical`
