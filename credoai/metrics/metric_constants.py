from credoai.metrics.credoai_metrics import (
    equal_opportunity_difference, false_discovery_rate, false_omission_rate
)
from functools import partial
from fairlearn import metrics as fl_metrics
from fairlearn.metrics._group_metric_set import BINARY_CLASSIFICATION_METRICS as fairlearn_binary
from sklearn import metrics as sk_metrics
from sklearn.metrics import SCORERS


# MODEL METRICS
BINARY_CLASSIFICATION_FUNCTIONS = {
    'false_positive_rate': fl_metrics.false_positive_rate,
    'false_negative_rate': fl_metrics.false_negative_rate,
    'false_discovery_rate': false_discovery_rate,
    'false_omission_rate': false_omission_rate,
    'true_positive_rate': fl_metrics.true_positive_rate,
    'true_negative_rate': fl_metrics.true_negative_rate,
    'precision_score': sk_metrics.precision_score,
    'accuracy_score': sk_metrics.accuracy_score,
    'balanced_accuracy_score': sk_metrics.balanced_accuracy_score,
    'matthews_correlation_coefficient': sk_metrics.matthews_corrcoef,
    'f1_score': sk_metrics.f1_score,
    'average_precision_score': sk_metrics.average_precision_score,
    'roc_auc_score': sk_metrics.roc_auc_score,
    'selection_rate': fl_metrics.selection_rate,
    'overprediction': fl_metrics._mean_overprediction,
    'underprediction': fl_metrics._mean_underprediction
}

# Define Fairness Metric Name Mapping
# Fairness metrics must have a similar signature to fairlearn.metrics.equalized_odds_difference
# (they should take sensitive_features and method)
fairness_metric_list = [fl_metrics.demographic_parity_difference,
                        fl_metrics.demographic_parity_ratio,
                        fl_metrics.equalized_odds_difference]
FAIRNESS_FUNCTIONS = {func.__name__: func for func in fairness_metric_list}
FAIRNESS_FUNCTIONS['equal_opportunity_difference'] = equal_opportunity_difference


# Define functions that require probabilities ***
PROBABILITY_FUNCTIONS = {"average_precision_score", "roc_auc_score"}

# *** Define Alternative Naming ***
METRIC_EQUIVALENTS = {
    'false_positive_rate': ['fpr', 'fallout_rate'],
    'false_negative_rate': ['fnr', 'miss_rate'], 
    'false_discovery_rate': ['fdr'], 
    'true_positive_rate': ['tpr', 'recall_score', 'recall', 'sensitivity', 'hit_rate'], 
    'true_negative_rate': ['tnr', 'specificity'], 
    'precision_score': ['precision'],
    'demographic_parity_difference': ['statistical_parity', 'demographic_parity'], 
    'demographic_parity_ratio': ['disparate_impact'], 
    'average_odds_difference': ['average_odds'], 
    'equal_opportunity_difference': ['equal_opportunity'], 
    'equalized_odds_difference': ['equalized_odds']
}

# DATASET METRICS
DATASET_METRIC_TYPES = [
    "sensitive_feature_prediction_score",
    "demographic_parity_ratio",
    "demographic_parity_difference"
]
