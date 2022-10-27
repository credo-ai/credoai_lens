import warnings
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd
from credoai.artifacts import TabularData
from credoai.evaluators import Evaluator
from credoai.evaluators.utils.validation import (
    check_artifact_for_nulls,
    check_data_instance,
    check_existence,
)
from credoai.evidence import MetricContainer
from credoai.utils.common import NotRunError, ValidationError, is_categorical
from credoai.utils.constants import MULTICLASS_THRESH
from credoai.utils.dataset_utils import ColumnTransformerUtil
from credoai.utils.model_utils import get_generic_classifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

METRIC_SUBSET = [
    "sensitive_feature-prediction_score",
    "demographic_parity-difference",
    "demographic_parity-ratio",
    "proxy_mutual_information-max",
]


class DataFairness(Evaluator):
    """Data Fairness for Credo AI.

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
    """

    def __init__(
        self,
        categorical_features_keys: Optional[List[str]] = None,
        categorical_threshold: float = 0.05,
    ):

        self.categorical_features_keys = categorical_features_keys
        self.categorical_threshold = categorical_threshold
        super().__init__()

    required_artifacts = {"data", "sensitive_feature"}

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

        return self

    def _validate_arguments(self):
        check_data_instance(self.data, TabularData)
        check_existence(self.data.sensitive_features, "sensitive_features")
        check_artifact_for_nulls(self.data, "Data")

        return self

    def evaluate(self):
        """
        Runs the assessment process.
        """
        res = {}
        ##  Aggregate results from all subprocess
        sensitive_feature_prediction_results = self._run_cv()
        mi_results = self._calculate_mutual_information()
        balance_metrics = self._assess_balance_metrics()
        # Note: Output for group difference is in a different format
        # Probably more suitable for table container. Address when this is
        # reintroduced in final results.
        group_differences = self._group_differences()

        res.update(
            {
                **balance_metrics,
                **sensitive_feature_prediction_results,
                **mi_results,
                "standardized_group_diffs": group_differences,
            }
        )

        # Select relevant results
        res = {k: v for k, v in res.items() if k in METRIC_SUBSET}

        # Reformat results
        res = [pd.DataFrame(v).assign(metric_type=k) for k, v in res.items()]
        res = pd.concat(res)
        res[["type", "subtype"]] = res.metric_type.str.split("-", expand=True)
        res.drop("metric_type", axis=1, inplace=True)

        self.results = [MetricContainer(res, **self.get_container_info())]
        return self

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
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            group_means = self.X.groupby(self.sensitive_features).mean()
        std = self.X.std(numeric_only=True)
        diffs = {}
        for group1, group2 in combinations(group_means.index, 2):
            diff = (group_means.loc[group1] - group_means.loc[group2]) / std
            diffs[f"{group1}-{group2}"] = diff.to_dict()
        return diffs

    def _run_cv(self):
        """
        Determines redundant encoding.

        A model is trained on the features to predict the sensitive attribute.
        The score, called "sensitive-feature-prediction-score" is a cross-validated ROC-AUC score.
        We scale the score from typical ROC range of 0.5-1 to 0-1.
        It quantifies the performance of this prediction.
        A high score means the data collectively serves as a proxy.

        Parameters
        ----------
        pipe : sklearn.pipeline
            Pipeline of transforms

        Returns
        -------
        ndarray
            Cross-validation score
        """
        results = {}
        if is_categorical(self.sensitive_features):
            sensitive_features = self.sensitive_features.cat.codes
        else:
            sensitive_features = self.sensitive_features

        pipe = self._make_pipe()
        scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class="ovo")
        n_folds = max(2, min(len(self.X) // 5, 5))
        cv_results = cross_val_score(
            pipe,
            self.X,
            sensitive_features,
            cv=StratifiedKFold(n_folds),
            scoring=scorer,
            error_score="raise",
        )

        # Get feature importances by running once
        pipe.fit(self.X, sensitive_features)
        model = pipe["model"]
        preprocessor = pipe["preprocessor"]
        col_names = ColumnTransformerUtil.get_ct_feature_names(preprocessor)
        feature_importances = pd.Series(
            model.feature_importances_, index=col_names
        ).sort_values(ascending=False)

        results["sensitive_feature-prediction_score"] = [
            {"value": max(cv_results.mean() * 2 - 1, 0)}
        ]  # move to 0-1 range

        # Reformat feature importance
        feature_importances = [
            {"feat_name": k, "value": v}
            for k, v in feature_importances.to_dict().items()
        ]
        results[
            "sensitive_feature-prediction_feature_importances"
        ] = feature_importances

        return results

    def _make_pipe(self):
        """
        Makes a pipeline.

        Returns
        -------
        sklearn.pipeline
            Pipeline of scaler and model transforms
        """
        categorical_features = self.categorical_features_keys.copy()
        numeric_features = [x for x in self.X.columns if x not in categorical_features]

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
        for name, column in self.X.iteritems():
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
        # Encode categorical features
        for col in self.categorical_features_keys:
            self.X[col] = self.X[col].astype("category").cat.codes

        discrete_features = [
            True if col in self.categorical_features_keys else False
            for col in self.X.columns
        ]

        # Use the right mutual information methods based on the feature type of the sensitive attribute
        if is_categorical(self.sensitive_features):
            sensitive_feature = self.sensitive_features.cat.codes
            mi = mutual_info_classif(
                self.X,
                sensitive_feature,
                discrete_features=discrete_features,
                random_state=42,
            )
            ref = mutual_info_classif(
                sensitive_feature.values[:, None],
                sensitive_feature,
                discrete_features=[True],
                random_state=42,
            )[0]
        else:
            mi = mutual_info_regression(
                self.X,
                self.sensitive_features,
                discrete_features=discrete_features,
                random_state=42,
            )
            ref = mutual_info_regression(
                self.sensitive_features.values[:, None],
                self.sensitive_features,
                random_state=42,
            )[0]

        # Normalize the mutual information values, if requested
        mi = pd.Series(mi, index=self.X.columns)
        if normalize:
            mi = mi / ref

        # Create the results
        mi = mi.sort_index().to_dict()
        mutual_information_results = []
        for k, v in mi.items():
            if k in self.categorical_features_keys:
                feature_type = "categorical"
            else:
                feature_type = "continuous"

            mutual_information_results.append(
                {
                    "feat_name": k,
                    "value": v,
                    "feature_type": feature_type,
                }
            )
        # Get max value
        max_proxy_value = max([i["value"] for i in mutual_information_results])

        return {
            "proxy_mutual_information": mutual_information_results,
            "proxy_mutual_information-max": [{"value": max_proxy_value}],
        }

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

        # Distribution of samples across groups
        sample_balance = (
            self.y.groupby(self.sensitive_features)
            .agg(
                count=(len),
                percentage=(lambda x: 100.0 * len(x) / len(self.y)),
            )
            .reset_index()
            .to_dict(orient="records")
        )
        balance_results["sample_balance"] = sample_balance

        # only calculate demographic parity and label balance when there are a reasonable
        # number of categories
        if len(self.y.unique()) < MULTICLASS_THRESH:
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                # Distribution of samples across groups
                label_balance = (
                    self.data.groupby([self.sensitive_features, self.y.name])
                    .size()
                    .unstack(fill_value=0)
                    .stack()
                    .reset_index(name="count")
                    .to_dict(orient="records")
                )
                balance_results["label_balance"] = label_balance

                # Fairness metrics
                r = (
                    self.data.groupby([self.sensitive_features, self.y.name])
                    .agg({self.y.name: "count"})
                    .groupby(level=0)
                    .apply(lambda x: x / float(x.sum()))
                    .rename({self.y.name: "ratio"}, inplace=False, axis=1)
                    .reset_index(inplace=False)
                )

            # Compute the maximum difference/ratio between any two pairs of groups

            def get_demo_parity(fun):
                return (
                    r.groupby(self.y.name)["ratio"]
                    .apply(fun)
                    .reset_index(name="value")
                    .iloc[1:]
                    .to_dict(orient="records")
                )

            balance_results["demographic_parity-difference"] = get_demo_parity(
                lambda x: np.max(x) - np.min(x)
            )
            balance_results["demographic_parity-ratio"] = get_demo_parity(
                lambda x: np.min(x) / np.max(x)
            )
        return balance_results
