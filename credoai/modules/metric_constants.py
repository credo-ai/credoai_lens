from functools import partial

from credoai.modules.credoai_metrics import (
    equal_opportunity_difference,
    false_discovery_rate,
    false_omission_rate,
    ks_statistic,
    gini_coefficient_discriminatory,
    population_stability_index,
)
from fairlearn import metrics as fl_metrics
from sklearn import metrics as sk_metrics

THRESHOLD_METRIC_CATEGORIES = ["BINARY_CLASSIFICATION_THRESHOLD"]

MODEL_METRIC_CATEGORIES = [
    "BINARY_CLASSIFICATION",
    "MULTICLASS_CLASSIFICATION",
    "REGRESSION",
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
    MODEL_METRIC_CATEGORIES + THRESHOLD_METRIC_CATEGORIES + NON_MODEL_METRIC_CATEGORIES
)

SCALAR_METRIC_CATEGORIES = MODEL_METRIC_CATEGORIES + NON_MODEL_METRIC_CATEGORIES

# MODEL METRICS
BINARY_CLASSIFICATION_FUNCTIONS = {
    "false_positive_rate": fl_metrics.false_positive_rate,
    "false_negative_rate": fl_metrics.false_negative_rate,
    "false_discovery_rate": false_discovery_rate,
    "false_omission_rate": false_omission_rate,
    "true_positive_rate": fl_metrics.true_positive_rate,
    "true_negative_rate": fl_metrics.true_negative_rate,
    "precision_score": sk_metrics.precision_score,
    "accuracy_score": sk_metrics.accuracy_score,
    "balanced_accuracy_score": sk_metrics.balanced_accuracy_score,
    "matthews_correlation_coefficient": sk_metrics.matthews_corrcoef,
    "f1_score": sk_metrics.f1_score,
    "average_precision_score": sk_metrics.average_precision_score,
    "roc_auc_score": sk_metrics.roc_auc_score,
    "selection_rate": fl_metrics.selection_rate,
    "overprediction": fl_metrics._mean_overprediction,
    "underprediction": fl_metrics._mean_underprediction,
    "gini_coefficient": gini_coefficient_discriminatory,
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
PROBABILITY_FUNCTIONS = {"average_precision_score", "roc_auc_score", "gini_coefficient"}

# *** Define Alternative Naming ***
METRIC_EQUIVALENTS = {
    "false_positive_rate": ["fpr", "fallout_rate"],
    "false_negative_rate": ["fnr", "miss_rate"],
    "false_discovery_rate": ["fdr"],
    "true_positive_rate": ["tpr", "recall_score", "recall", "sensitivity", "hit_rate"],
    "true_negative_rate": ["tnr", "specificity"],
    "precision_score": ["precision"],
    "demographic_parity_difference": ["statistical_parity", "demographic_parity"],
    "demographic_parity_ratio": ["disparate_impact"],
    "average_odds_difference": ["average_odds"],
    "equal_opportunity_difference": ["equal_opportunity"],
    "equalized_odds_difference": ["equalized_odds"],
    "mean_absolute_error": ["MAE"],
    "mean_squared_error": ["MSE", "MSD", "mean_squared_deviation"],
    "root_mean_squared_error": ["RMSE"],
    "r2_score": ["r_squared", "r2"],
    "gini_coefficient": [
        "gini_index",
        "discriminatory_gini_index",
        "discriminatory_gini",
    ],
    "population_stability_index": ["psi", "PSI"],
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
