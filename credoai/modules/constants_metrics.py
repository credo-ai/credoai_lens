"""Constants for threshold metrics

Define relationships between metric names (strings) and
metric functions, as well as alternate names for each metric name
"""

from functools import partial

from fairlearn import metrics as fl_metrics
from sklearn import metrics as sk_metrics

from credoai.artifacts.model.constants_model import MODEL_TYPES
from credoai.modules.metrics_credoai import (
    equal_opportunity_difference,
    false_discovery_rate,
    false_omission_rate,
    gini_coefficient_discriminatory,
    ks_statistic,
    ks_statistic_binary,
    multiclass_confusion_metrics,
)

THRESHOLD_METRIC_CATEGORIES = ["BINARY_CLASSIFICATION_THRESHOLD"]

MODEL_METRIC_CATEGORIES = [
    "CLUSTERING",
    "FAIRNESS",
] + THRESHOLD_METRIC_CATEGORIES

NON_MODEL_METRIC_CATEGORIES = [
    "PRIVACY",
    "SECURITY",
    "DATASET",
    "CUSTOM",
]

METRIC_CATEGORIES = (
    MODEL_TYPES
    + MODEL_METRIC_CATEGORIES
    + THRESHOLD_METRIC_CATEGORIES
    + NON_MODEL_METRIC_CATEGORIES
)

SCALAR_METRIC_CATEGORIES = MODEL_METRIC_CATEGORIES + NON_MODEL_METRIC_CATEGORIES

# MODEL METRICS
# Define Binary classification name mapping.
# Binary classification metrics must have a similar signature to sklearn metrics
BINARY_CLASSIFICATION_FUNCTIONS = {
    "accuracy_score": sk_metrics.accuracy_score,
    "average_precision_score": sk_metrics.average_precision_score,
    "balanced_accuracy_score": sk_metrics.balanced_accuracy_score,
    "f1_score": sk_metrics.f1_score,
    "false_discovery_rate": false_discovery_rate,
    "false_negative_rate": fl_metrics.false_negative_rate,
    "false_omission_rate": false_omission_rate,
    "false_positive_rate": fl_metrics.false_positive_rate,
    "gini_coefficient": gini_coefficient_discriminatory,
    "matthews_correlation_coefficient": sk_metrics.matthews_corrcoef,
    "overprediction": fl_metrics._mean_overprediction,
    "precision_score": sk_metrics.precision_score,
    "roc_auc_score": sk_metrics.roc_auc_score,
    "selection_rate": fl_metrics.selection_rate,
    "true_negative_rate": fl_metrics.true_negative_rate,
    "true_positive_rate": fl_metrics.true_positive_rate,
    "underprediction": fl_metrics._mean_underprediction,
    "ks_score_binary": ks_statistic_binary,
}

# Define Multiclass classification name mapping.
# Multiclass classification metrics must have a similar signature to sklearn metrics
MULTICLASS_CLASSIFICATION_FUNCTIONS = {
    "accuracy_score": partial(multiclass_confusion_metrics, metric="ACC"),
    "balanced_accuracy_score": sk_metrics.balanced_accuracy_score,
    "f1_score": partial(sk_metrics.f1_score, average="weighted"),
    "false_discovery_rate": partial(multiclass_confusion_metrics, metric="FDR"),
    "false_negative_rate": partial(multiclass_confusion_metrics, metric="FNR"),
    "false_positive_rate": partial(multiclass_confusion_metrics, metric="FPR"),
    "gini_coefficient": partial(
        gini_coefficient_discriminatory, multi_class="ovo", average="weighted"
    ),
    "matthews_correlation_coefficient": sk_metrics.matthews_corrcoef,
    "overprediction": fl_metrics._mean_overprediction,
    "precision_score": partial(sk_metrics.precision_score, average="weighted"),
    "roc_auc_score": partial(
        sk_metrics.roc_auc_score, multi_class="ovo", average="weighted"
    ),
    "selection_rate": fl_metrics.selection_rate,
    "true_negative_rate": partial(multiclass_confusion_metrics, metric="TNR"),
    "true_positive_rate": partial(multiclass_confusion_metrics, metric="TPR"),
    "underprediction": fl_metrics._mean_underprediction,
}

# Define Fairness Metric Name Mapping
# Fairness metrics must have a similar signature to fairlearn.metrics.equalized_odds_difference
# (they should take sensitive_features and method)
FAIRNESS_FUNCTIONS = {
    "demographic_parity_difference": fl_metrics.demographic_parity_difference,
    "demographic_parity_ratio": fl_metrics.demographic_parity_ratio,
    "equalized_odds_difference": fl_metrics.equalized_odds_difference,
    "equal_opportunity_difference": equal_opportunity_difference,
}


# Define functions that require probabilities ***
PROBABILITY_FUNCTIONS = {
    "average_precision_score",
    "roc_auc_score",
    "gini_coefficient",
    "ks_score_binary",
}

# *** Define Alternative Naming ***
METRIC_EQUIVALENTS = {
    "average_odds_difference": ["average_odds"],
    "average_precision_score": ["average_precision"],
    "demographic_parity_difference": ["statistical_parity", "demographic_parity"],
    "demographic_parity_ratio": ["disparate_impact"],
    "equal_opportunity_difference": ["equal_opportunity"],
    "equalized_odds_difference": ["equalized_odds"],
    "false_positive_rate": ["fpr", "fallout_rate", "false_match_rate"],
    "false_negative_rate": ["fnr", "miss_rate", "false_non_match_rate"],
    "false_discovery_rate": ["fdr"],
    "gini_coefficient": [
        "gini_index",
        "discriminatory_gini_index",
        "discriminatory_gini",
    ],
    "mean_absolute_error": ["MAE"],
    "mean_squared_error": ["MSE", "MSD", "mean_squared_deviation"],
    "population_stability_index": ["psi", "PSI"],
    "precision_score": ["precision"],
    "root_mean_squared_error": ["RMSE"],
    "r2_score": ["r_squared", "r2"],
    "true_positive_rate": ["tpr", "recall_score", "recall", "sensitivity", "hit_rate"],
    "true_negative_rate": ["tnr", "specificity"],
    "target_ks_statistic": ["ks_score_regression", "ks_score"],
    "ks_score_binary": ["ks_score"],
}

# DATASET METRICS
DATASET_METRIC_TYPES = [
    "sensitive_feature_prediction_score",
    "demographic_parity_ratio",
    "demographic_parity_difference",
    "max_proxy_mutual_information",
]

# PRIVACY METRICS
PRIVACY_METRIC_TYPES = [
    "rule_based_attack_score",
    "model_based_attack_score",
    "membership_inference_attack_score",
]

# SECURITY METRICS
SECURITY_METRIC_TYPES = ["extraction_attack_score", "evasion_attack_score"]

# REGRESSION METRICS
REGRESSION_FUNCTIONS = {
    "explained_variance_score": sk_metrics.explained_variance_score,
    "max_error": sk_metrics.max_error,
    "mean_absolute_error": sk_metrics.mean_absolute_error,
    "mean_squared_error": sk_metrics.mean_squared_error,
    "root_mean_squared_error": partial(sk_metrics.mean_squared_error, squared=False),
    "mean_squared_log_error": sk_metrics.mean_squared_log_error,
    "mean_absolute_percentage_error": sk_metrics.mean_absolute_percentage_error,
    "median_absolute_error": sk_metrics.median_absolute_error,
    "r2_score": sk_metrics.r2_score,
    "mean_poisson_deviance": sk_metrics.mean_poisson_deviance,
    "mean_gamma_deviance": sk_metrics.mean_gamma_deviance,
    "d2_tweedie_score": sk_metrics.d2_tweedie_score,
    "mean_pinball_loss": sk_metrics.mean_pinball_loss,
    "target_ks_statistic": ks_statistic,
}
