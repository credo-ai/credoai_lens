import numpy as np
import pandas as pd
from sklearn import feature_extraction, feature_selection, impute, pipeline


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
        if hasattr(estimator, "get_feature_names_out"):
            if isinstance(estimator, feature_extraction.text._VectorizerMixin):
                # handling all vectorizers
                return [f"vec_{f}" for f in estimator.get_feature_names_out()]
            else:
                return estimator.get_feature_names_out(feature_in)
        elif hasattr(estimator, "get_feature_names"):
            return estimator.get_feature_names(feature_in)
        elif isinstance(estimator, feature_selection._base.SelectorMixin):
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
            if name != "remainder":
                if isinstance(estimator, pipeline.Pipeline):
                    current_features = features
                    for step in estimator:
                        current_features = ColumnTransformerUtil.get_feature_out(
                            step, current_features
                        )
                    features_out = current_features
                else:
                    features_out = ColumnTransformerUtil.get_feature_out(
                        estimator, features
                    )
                output_features.extend(features_out)
            elif estimator == "passthrough":
                output_features.extend(ct._feature_names_in[features])
        return output_features


def scrub_data(credo_data, nan_strategy="ignore"):
    """Return scrubbed data

    Implements NaN strategy indicated by nan_strategy before returning
    X, y and sensitive_features dataframes/series.

    Parameters
    ----------
    credo_data : CredoData
        Data object
    nan_strategy : str or callable, optional
        The strategy for dealing with NaNs.

        -- If "ignore" do nothing,
        -- If "drop" drop any rows with any NaNs. X must be a pd.DataFrame
        -- If any other string, pass to the "strategy" argument of `Simple Imputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html>`_.

        You can also supply your own imputer with
        the same API as `SimpleImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html>`_.

    Returns
    -------
    X

    Raises
    ------
    ValueError
        ValueError raised for nan_strategy cannot be used by SimpleImputer
    """
    if credo_data.X_type not in (pd.DataFrame, np.ndarray):
        return credo_data
    X, y, sensitive_features = credo_data.get_data().values()
    imputed = None
    if nan_strategy == "drop":
        if credo_data.X_type == pd.DataFrame:
            # determine index of no nan rows
            tmp = pd.concat([X, y, sensitive_features], axis=1).dropna()
            # apply dropped index
            X = X.loc[tmp.index]
            if y is not None:
                y = y.loc[tmp.index]
            if sensitive_features is not None:
                sensitive_features = sensitive_features.loc[tmp.index]
        else:
            raise TypeError("X must be a pd.DataFrame when using the drop option")
    elif nan_strategy == "ignore":
        pass
    elif isinstance(nan_strategy, str):
        try:
            imputer = impute.SimpleImputer(strategy=nan_strategy)
            imputed = imputer.fit_transform(X)
        except ValueError:
            raise ValueError(
                "Nan_strategy could not be successfully passed to SimpleImputer as a 'strategy' argument"
            )
    else:
        imputed = nan_strategy.fit_transform(X)
    if imputed:
        X = X.copy()
        X.iloc[:, :] = imputed
    return X, y, sensitive_features
