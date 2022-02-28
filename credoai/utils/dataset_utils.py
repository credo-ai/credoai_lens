import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer

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

class ColumnTransformerUtil:
    """Utility functions for ColumnTransformer

    ColumnTransformer is a helpful preprocessing utility from sklearn.
    However, it makes getting the original feature names difficult, which
    makes interpreting feature importance hard. This utility class
    defined a `get_ct_feature_names` function which takes in a 
    ColumnTransformer instance and outputs a list of feature names

    Ref: https://stackoverflow.com/a/57534118
    """
    @staticmethod
    def get_feature_out(estimator, feature_in):
        if hasattr(estimator,'get_feature_names'):
            if isinstance(estimator, _VectorizerMixin):
                # handling all vectorizers
                return [f'vec_{f}' \
                    for f in estimator.get_feature_names()]
            else:
                return estimator.get_feature_names(feature_in)
        elif isinstance(estimator, SelectorMixin):
            return np.array(feature_in)[estimator.get_support()]
        else:
            return feature_in

    @staticmethod
    def get_ct_feature_names(ct):
        # handles all estimators, pipelines inside ColumnTransfomer
        # doesn't work when remainder =='passthrough'
        # which requires the input column names.
        output_features = []

        for name, estimator, features in ct.transformers_:
            if name!='remainder':
                if isinstance(estimator, Pipeline):
                    current_features = features
                    for step in estimator:
                        current_features = ColumnTransformerUtil.get_feature_out(step, current_features)
                    features_out = current_features
                else:
                    features_out = ColumnTransformerUtil.get_feature_out(estimator, features)
                output_features.extend(features_out)
            elif estimator=='passthrough':
                output_features.extend(ct._feature_names_in[features])
        return output_features