import re
from credoai.utils.common import remove_suffix

# *** Define Alternative Naming ***

METRIC_EQUIVALENTS = {
    'false_positive_rate': ['fpr', 'fallout_rate'],
    'false_negative_rate': ['fnr', 'miss_rate'], 
    'false_discovery_rate': ['fdr'], 
    'true_positive_rate': ['tpr', 'recall_score', 'sensitivity', 'hit_rate'], 
    'true_negative_rate': ['tnr', 'specificity'], 
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

def standardize_metric_name(metric):
    # standardize
    # lower, remove spaces, replace delimiters with underscores
    standard =  '_'.join(re.split('[- \s _]', 
                         re.sub('\s\s+', ' ', metric.lower())))
    standard = remove_suffix(remove_suffix(standard, '_difference'), '_parity')
    return STANDARD_CONVERSIONS.get(standard, standard)

def humanize_metric_name(metric):
    return ' '.join(metric.split('_')).title()