import pandas as pd

def concat_features_label_to_dataframe(X, y, sensitive_features):
    """A utility method that concatenates all features and labels into a single dataframe

    Returns
    -------
    pandas.dataframe, str, str
        Full dataset dataframe, sensitive feature name, label name
    """
    if isinstance(sensitive_features, pd.Series):
        df = pd.concat([X, sensitive_features], axis=1)
        sensitive_feature_name = sensitive_features.name
    else:
        df = X.copy()
        df['sensitive_feature'] = sensitive_features
        sensitive_feature_name = 'sensitive_feature'

    if isinstance(y, pd.Series):
        df = pd.concat([df, y], axis=1)
        label_name = y.name
    else:
        label_name = 'label'
        df[label_name] = sensitive_features

    return df, sensitive_feature_name, label_name