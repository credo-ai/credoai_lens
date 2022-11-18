
Ranking fairness
================


Ranking fairness evaluator for Credo AI

This module takes in ranking results and provides functionality to perform fairness assessment
    The results should include rankings, sensitive features, and optionally, scores.

skew_parity_difference: max_skew - min_skew, where skew is the proportion of the selected
    items from a group over the desired proportion for that group.
    It ranges from 0 to inf and the ideal value is 0.
ndkl: is a metric that accounts for increasing ranks. It is non-negative, with larger values
    indicating a greater divergence between the desired and actual distributions of
    sensitive attribute labels.
    It ranges from 0 to inf and the ideal value is 0.
demographic_parity_ratio: min_selection_rate / max_selection_rate, where selection rate
    is the proportion of the selected items from a group over the number of items for
    that group in the pool.
    It ranges from 0 to 1 and ideal value is 1.
balance_ratio: min_presence / max_presence, where presence is the number of the selected items
    from a group.
    It ranges from 0 to 1 and ideal value is 1.
qualified_demographic_parity_ratio: demographic_parity_ratio but with a qualified (i.e., score
    greater than or equal to q) filter applied to the items.
    It ranges from 0 to 1 and ideal value is 1.
qualified_balance_ratio: balance_ratio but with a qualified (i.e., score greater than or equal
    to q) filter applied to the items.
    It ranges from 0 to 1 and ideal value is 1.
calibrated_demographic_parity_ratio: demographic_parity_ratio but with the selected set from
    specified score bins. This is to audit if items with similiar scores are are treated similarly
    (via proportional presence) regardless of group membership.
    It ranges from 0 to 1 and ideal value is 1.
calibrated_balance_ratio: balance_ratio but with the selected set from
    specified score bins. This is to audit if items with similiar scores are are treated similarly
    (via equal presence) regardless of group membership.
    It ranges from 0 to 1 and ideal value is 1.
relevance_parity_ratio: to audit if groups are represented proportional to their average score
    (i.e., score-based relevance)
    It ranges from 0 to 1 and ideal value is 1.
score_parity_ratio:  min_average_Score / max_average_Score, where average score
    is the average score of the selected items from a group.
    It ranges from 0 to 1 and ideal value is 1.
score_balance_ratio: min_total_Score / max_total_Score, where total score
    is the total score of the selected items from a group.
    It ranges from 0 to 1 and ideal value is 1.
score_empirical_distribution: score empirical distributions for each demographic group as tables
    The x axis is scores and the y axis is cumulative probabilities (ranges from 0 to 1)
    It is useful for a visual examination of the distribution of scores for the different groups.

Parameters
----------
sensitive_features : pandas.Series
    A series of the sensitive feature labels (e.g., "male", "female") which should
    be used to create subgroups
rankings : pandas.Series of type int
    The computed ranks
    It should be passed to TabularData's y argument with the column name `rankings`
scores : pandas.Series of type int or float, Optional
    A series of the scores
    It should be passed to TabularData's y argument with the column name `scores`
k: int, Optional
    The top k items are considered as the selected subset
    If not provided, the top 50% of the items are considered as selected
q: float, Optional
    The relevance score for which items in the pool that have score >= q are "relevant".
    These two metrics require this to be provided: `qualified_demographic_parity_ratio`
    and `qualified_balance_ratio`
lb_bin: numpy array of shape = (n_bins), Optional
    The lower bound scores for each bin (bin is greater than or equal to lower bound).
    These two metrics require this to be provided: `calibrated_demographic_parity_ratio`
    and `calibrated_balance_ratio`
ub_bin: numpy array of shape = (n_bins), Optional
    The upper bound scores for each bin (bin is less than upper bound).
    These two metrics require this to be provided: `calibrated_demographic_parity_ratio`
    and `calibrated_balance_ratio`
desired_proportions: dict, Optional
    The desired proportion for each subgroups (e.g., {"male":0.4, "female":0.6})
    If not provided, equal proportions are used for calculation of `skew` score
down_sampling_step : int, optional
    down-sampling step for scores empirical distribution curve
    If not provided, down-sampling is done such that the curve length be nearly 100
