from credoai.utils.metrics import (
    equal_opportunity_difference, false_discovery_rate, false_omission_rate
)
from functools import partial
from fairlearn import metrics as fl_metrics
from fairlearn.metrics._group_metric_set import BINARY_CLASSIFICATION_METRICS as fairlearn_binary
from sklearn import metrics as sk_metrics
from sklearn.metrics import SCORERS

# *** CONSTANTS ***
# *** Define Basic Metric Name Mapping ***

BINARY_CLASSIFICATION_METRICS = {
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
fairness_metric_list = [fl_metrics.demographic_parity_difference,
                        fl_metrics.demographic_parity_ratio,
                        fl_metrics.equalized_odds_difference]
FAIRNESS_METRICS = {func.__name__: func for func in fairness_metric_list}
FAIRNESS_METRICS['equal_opportunity_difference'] = equal_opportunity_difference

# Define functions that require probabilities ***
PROBABILITY_METRICS = {"average_precision_score", "roc_auc_score"}

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

STANDARD_CONVERSIONS = {}
for standard, equivalents in METRIC_EQUIVALENTS.items():
    conversions = {name: standard for name in equivalents}
    STANDARD_CONVERSIONS.update(conversions)
    
# *** collate all metrics ***
ALL_METRICS = list(BINARY_CLASSIFICATION_METRICS.keys()) + \
    list(FAIRNESS_METRICS.keys()) + list(STANDARD_CONVERSIONS.keys())

# humanize metric names
def humanize_metric_name(metric):
    return ' '.join(metric.split('_')).title()
ALL_METRICS_HUMANIZED = [humanize_metric_name(m) for m in ALL_METRICS]

