import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd
from connect.evidence import MetricContainer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credoai.artifacts import TabularData
from credoai.evaluators.evaluator import Evaluator
from credoai.evaluators.utils.validation import (
    check_data_for_nulls,
    check_data_instance,
    check_existence,
)
from credoai.utils.common import NotRunError, ValidationError, is_categorical
from credoai.utils.constants import MULTICLASS_THRESH
from credoai.utils.dataset_utils import ColumnTransformerUtil
from credoai.utils.model_utils import get_generic_classifier

METRIC_SUBSET = [
    "sensitive_feature-prediction_score",
    "demographic_parity-difference",
    "demographic_parity-ratio",
    "proxy_mutual_information-max",
]


class DataFairness(Evaluator):
    """
    Data Fairness evaluator for Credo AI (Experimental)

    This evaluator performs a fairness evaluation on the dataset. Given a sensitive feature,
    it calculates a number of assessments:

    - group differences of features
    - evaluates whether features in the dataset are proxies for the sensitive feature
    - whether the entire dataset can be seen as a proxy for the sensitive feature
    (i.e., the sensitive feature is "redundantly encoded")

    Required Artifacts
    ------------------
        **Required Artifacts**

        Generally artifacts are passed directly to :class:`credoai.lens.Lens`, which
        handles evaluator setup. However, if you are using the evaluator directly, you
        will need to pass the following artifacts when instantiating the evaluator:

        - data: :class:`credoai.artifacts.TabularData`
            The data to evaluate, which must include a sensitive feature

    Parameters
    ----------
    categorical_features_keys : list[str], optional
        Names of the categorical features
    categorical_threshold : float
        Parameter for automatically identifying categorical columns. See
        :class:`credoai.utils.common.is_categorical` for more details.
    """

    required_artifacts = {"data", "sensitive_feature"}

    def __init__(
        self,
        categorical_features_keys: Optional[List[str]] = None,
        categorical_threshold: float = 0.05,
    ):
        self.categorical_features_keys = categorical_features_keys
        self.categorical_threshold = categorical_threshold
        super().__init__()

    def _validate_arguments(self):
        check_data_instance(self.data, TabularData)
        check_existence(self.data.sensitive_features, "sensitive_features")
        check_data_for_nulls(self.data, "Data")

    def _setup(self):
        self.data_to_eval = self.data  # Pick the only member

        self.sensitive_features = self.data_to_eval.sensitive_feature
        self.data = pd.concat([self.data_to_eval.X, self.data_to_eval.y], axis=1)
        self.X = self.data_to_eval.X
        self.y = self.data_to_eval.y

        # set up categorical features
        if self.categorical_features_keys:
            for sensitive_feature_name in self.sensitive_features:
                if sensitive_feature_name in self.categorical_features_keys:
                    self.sensitive_features[
                        sensitive_feature_name
                    ] = self.sensitive_features[sensitive_feature_name].astype(
                        "category"
                    )
                    self.categorical_features_keys.remove(sensitive_feature_name)
        else:
            self.categorical_features_keys = self._find_categorical_features(
                self.categorical_threshold
            )

        # Encode categorical features
        for col in self.categorical_features_keys:
            self.X[col] = self.X[col].astype("category").cat.codes

        # Locate discrete features in dataset
        self.discrete_features = [
            True if col in self.categorical_features_keys else False
            for col in self.X.columns
        ]

        return self

    def evaluate(self):
        """
        Runs the assessment process.
        """
        ##  Aggregate results from all subprocess
        sensitive_feature_prediction_results = self._check_redundant_encoding()

        mi_results = self._calculate_mutual_information()
        balance_metrics = self._assess_balance_metrics()
        group_differences = self._group_differences()

        # Format the output
        self.results = self._format_results(
            sensitive_feature_prediction_results,
            mi_results,
            balance_metrics,
            group_differences,
        )
        return self

    def _check_redundant_encoding(self):
        """Assesses whether the dataset 'redundantly encodes' the sensitive feature

        Reundant encoding means that information of the sensitive feature is embedded in the
        rest of the dataset. This means that sensitive feature information is contained
        and therefore "leakable" to the trained model. This is a more robust measure of
        sensitive feature proxy than looking at a single proxy feature, as information of
        the sensitive feature may not exist on one dataset feature, but rather in the confluence
        of many.

        Reundant encoding is measured by performing a feature inference attack using the entire
        dataset
        """
        results = FeatureInference()(
            self.X, self.sensitive_features, self.categorical_features_keys
        )
        results = {f"sensitive_feature_inference_{k}": v for k, v in results.items()}
        return results

    def _format_results(
        self,
        sensitive_feature_prediction_results,
        mi_results,
        balance_metrics,
        group_differences,
    ):
        """
        Formats the results into a dataframe for MetricContainer

        Parameters
        ----------
        sensitive_feature_prediction_results : dict
            Results of redundant encoding calculation
        mi_results : dict
            Results of mutual information calculation
        balance_metrics : dict
            Results of balanced statistics calculation
        group_differences : dict
            Results of standardized difference calculation
        """
        res = {
            **balance_metrics,
            **sensitive_feature_prediction_results,
            **mi_results,
            **group_differences,
        }

        # Select relevant results
        res = {k: v for k, v in res.items() if k in METRIC_SUBSET}

        # Reformat results
        res = [pd.DataFrame(v).assign(metric_type=k) for k, v in res.items()]
        res = pd.concat(res)
        res[["type", "subtype"]] = res.metric_type.str.split("-", expand=True)
        res.drop("metric_type", axis=1, inplace=True)

        return [MetricContainer(res, **self.get_info())]

    def _group_differences(self):
        """
        Calculates standardized mean differences.

        It is performed for all numeric features and all possible group pairs combinations present in the sensitive feature.

        Returns
        -------
        dict, nested
            Key: sensitive feature groups pair
            Values: dict
                Key: name of feature
                Value: standardized mean difference
        """
        group_means = self.X.groupby(self.sensitive_features).mean()
        std = self.X.std(numeric_only=True)
        diffs = {}
        for group1, group2 in combinations(group_means.index, 2):
            diff = (group_means.loc[group1] - group_means.loc[group2]) / std
            diffs[f"{group1}-{group2}"] = diff.to_dict()
        diffs = {"standardized_group_diffs": diffs}
        return diffs

    def _find_categorical_features(self, threshold):
        """
        Identifies categorical features.

        Returns
        -------
        list
            Names of categorical features
        """
        if is_categorical(self.sensitive_features, threshold=threshold):
            self.sensitive_features = self.sensitive_features.astype("category")
        cat_cols = []
        for name, column in self.X.items():
            if is_categorical(column, threshold=threshold):
                cat_cols.append(name)
        return cat_cols

    def _calculate_mutual_information(self, normalize=True):
        """
        Calculates normalized mutual information between sensitive feature and other features.

        Mutual information is the "amount of information" obtained about the sensitive feature by observing another feature.
        Mutual information is useful to proxy detection purposes.

        Parameters
        ----------
        normalize : bool, optional
            If True, calculated mutual information values are normalized
            Normalization is done via dividing by the mutual information between the sensitive feature and itself.

        Returns
        -------
        dict, nested
            Key: feature name
            Value: mutual information and considered feature type (categorical/continuous)
        """

        # Use the right mutual information methods based on the feature type of the sensitive attribute
        if is_categorical(self.sensitive_features):
            mi, ref = self._categorical_mi()
        else:
            mi, ref = self._numerical_mi()

        # Normalize the mutual information values, if requested
        mi = pd.Series(mi, index=self.X.columns)
        if normalize:
            mi = mi / ref

        # Create the results
        mi = mi.sort_index().to_dict()
        mutual_information_results = [
            {
                "feat_name": k,
                "value": v,
                "feature_type": "categorical"
                if k in self.categorical_features_keys
                else "continuous",
            }
            for k, v in mi.items()
        ]

        # Get max value
        max_proxy_value = max([i["value"] for i in mutual_information_results])

        return {
            "proxy_mutual_information": mutual_information_results,
            "proxy_mutual_information-max": [{"value": max_proxy_value}],
        }

    def _numerical_mi(self):
        """Calculate mutual information for numerical features"""
        mi = mutual_info_regression(
            self.X,
            self.sensitive_features,
            discrete_features=self.discrete_features,
            random_state=42,
        )
        ref = mutual_info_regression(
            self.sensitive_features.values[:, None],
            self.sensitive_features,
            random_state=42,
        )[0]

        return mi, ref

    def _categorical_mi(self):
        """
        Calculate mutual information for categorical features
        """
        sensitive_feature = self.sensitive_features.cat.codes
        mi = mutual_info_classif(
            self.X,
            sensitive_feature,
            discrete_features=self.discrete_features,
            random_state=42,
        )
        ref = mutual_info_classif(
            sensitive_feature.values[:, None],
            sensitive_feature,
            discrete_features=[True],
            random_state=42,
        )[0]
        return mi, ref

    def _assess_balance_metrics(self):
        """
        Calculate dataset balance statistics and metrics.

        Returns
        -------
        dict
            'sample_balance': distribution of samples across groups
            'label_balance': distribution of labels across groups
            'metrics': demographic parity difference and ratio between groups for all preferred label value possibilities
        """
        balance_results = {}

        from collections import Counter

        # Distribution of samples across groups
        sens_feat_breakdown = Counter(self.sensitive_features)
        total = len(self.sensitive_features)
        balance_results["sample_balance"] = [
            {"race": k, "count": v, "percentage": v * 100 / total}
            for k, v in sens_feat_breakdown.items()
        ]

        # only calculate demographic parity and label balance when there are a reasonable
        # number of categories
        if len(self.y.unique()) < MULTICLASS_THRESH:
            # Distribution of samples across groups
            sens_feat_label = self.data[[self.y.name]]
            sens_feat_label[self.sensitive_features.name] = self.sensitive_features
            label_balance = sens_feat_label.value_counts().reset_index(name="count")

            balance_results["label_balance"] = label_balance.to_dict(orient="records")

            # Fairness metrics
            ## Get Ratio of total
            label_balance["ratio"] = label_balance["count"] / label_balance.groupby(
                self.sensitive_features.name
            )["count"].transform("sum")
            sens_feat_y_counts = label_balance.drop("count", axis=1)

            # Compute the maximum difference/ratio between any two pairs of groups
            balance_results["demographic_parity-difference"] = get_demographic_parity(
                sens_feat_y_counts, self.y.name, "difference"
            )
            balance_results["demographic_parity-ratio"] = get_demographic_parity(
                sens_feat_y_counts, self.y.name, "ratio"
            )
        return balance_results


############################################
## Evaluation helper functions

## Helper functions create evidences
## to be passed to .evaluate to be wrapped
## by evidence containers
############################################


class FeatureInference:
    def __init__(self):
        """
        Class to infer a particular feature

        A model is trained on the X features to predict the target.
        The prediction is a cross-validated ROC-AUC score.
        We scale the score from typical ROC range of 0.5-1 to 0-1.
        It quantifies the performance of this prediction.
        A high score means the data collectively serves as a proxy for the target.
        """

    def __call__(
        self, X: pd.DataFrame, target: pd.Series, categorical_features_keys: pd.Series
    ):
        """
        Performs feature inference attack

        Parameters
        ----------
        X : pd.DataFrame
            Dataset used for the assessment
        target : pd.Series
            Feature we are trying to infer from X. In the evaluator this is sensitive features.
        categorical_features_keys : pd.Series
            Series describing which are the categorical variables in X

        Returns
        -------
        dict
            Nested dictionary with all the results
        """

        results = {}
        if is_categorical(target):
            target = target.cat.codes
        else:
            target = target
        pipe = self._make_pipe(X, categorical_features_keys)

        results = {
            "scaled_ROC_score": [{"value": self._pipe_scores(pipe, X, target)}],
            "feature_importances": self._pipe_importance(pipe, X, target),
        }
        return results

    def _make_pipe(
        self, X: pd.DataFrame, categorical_features_keys: pd.Series
    ) -> Pipeline:
        """
        Makes a pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset used for the assessment
        categorical_features_keys : pd.Series
            Series describing which are the categorical variables in X

        Returns
        -------
        sklearn.pipeline
            Pipeline of scaler and model transforms
        """
        categorical_features = categorical_features_keys.copy()
        numeric_features = [x for x in X.columns if x not in categorical_features]

        # Define features tansformers
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        transformers = []
        if len(categorical_features):
            categorical_transformer = OneHotEncoder(handle_unknown="ignore")
            transformers.append(("cat", categorical_transformer, categorical_features))
        if len(numeric_features):
            numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
            transformers.append(("num", numeric_transformer, numeric_features))
        preprocessor = ColumnTransformer(transformers=transformers)

        model = get_generic_classifier()

        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        return pipe

    def _pipe_importance(self, pipe, X, target):
        """Gets feature importances for pipeline"""
        # Get feature importances by running once
        pipe.fit(X, target)
        model = pipe["model"]
        preprocessor = pipe["preprocessor"]
        col_names = ColumnTransformerUtil.get_ct_feature_names(preprocessor)
        feature_importances = pd.Series(
            model.feature_importances_, index=col_names
        ).sort_values(ascending=False)

        # Reformat feature importance
        feature_importances = [
            {"feat_name": k, "value": v}
            for k, v in feature_importances.to_dict().items()
        ]
        return feature_importances

    def _pipe_scores(self, pipe, X, target):
        """Calculates average cross-validated scores for pipeline"""
        scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class="ovo")
        n_folds = max(2, min(len(X) // 5, 5))
        cv_results = cross_val_score(
            pipe,
            X,
            target,
            cv=StratifiedKFold(n_folds),
            scoring=scorer,
            error_score="raise",
        )
        return max(cv_results.mean() * 2 - 1, 0)


def get_demographic_parity(df: pd.DataFrame, target_name: str, fun: str) -> dict:
    """
    Calculates maximum difference/ratio between target categories.

    Parameters
    ----------
    df : pd.DataFrame
        Data grouped by sensitive feature and y, then  count of y over total
        instances is calculated.
    target_name : str
        Name of y object
    fun : str
        Either "difference" or "ratio": indicates which calculation to perform.

    Returns
    -------
    dict
        _description_
    """
    funcs = {
        "difference": lambda x: np.max(x) - np.min(x),
        "ratio": lambda x: np.min(x) / np.max(x),
    }
    return (
        df.groupby(target_name)["ratio"]
        .apply(funcs[fun])
        .reset_index(name="value")
        .iloc[1:]
        .to_dict(orient="records")
    )
