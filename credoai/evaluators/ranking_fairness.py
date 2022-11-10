"""Ranking Fairness evaluator"""
import math
from collections import Counter

import numpy as np
import pandas as pd
from credoai.artifacts import TabularData
from credoai.evaluators import Evaluator
from credoai.evaluators.utils.validation import (
    check_artifact_for_nulls,
    check_data_instance,
    check_existence,
    check_feature_presence,
)
from credoai.evidence.containers import MetricContainer, TableContainer
from credoai.utils.common import ValidationError
from credoai.utils.dataset_utils import empirical_distribution_curve
from finsfairauditing import fins

EPSILON = 1e-12
METRIC_SUBSET = [
    "skew_parity_difference-score",
    "ndkl-score",
    "demographic_parity_ratio-score",
    "balance_ratio-score",
    "qualified_demographic_parity_ratio-score",
    "qualified_balance_ratio-score",
    "calibrated_demographic_parity_ratio-score",
    "calibrated_balance_ratio-score",
    "relevance_parity_ratio-score",
    "score_parity_ratio-score",
    "score_balance_ratio-score",
]


class RankingFairness(Evaluator):
    """Ranking fairness evaluator for Credo AI

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
    """

    def __init__(
        self,
        k: int = None,
        q: float = None,
        lb_bin: list = None,
        ub_bin: list = None,
        desired_proportions: dict = None,
        down_sampling_step: int = None,
    ):
        self.desired_proportions = (desired_proportions,)
        self.k = k
        self.q = q
        self.down_sampling_step = down_sampling_step
        if lb_bin is not None and ub_bin is not None:
            self.lb_bin = np.array(lb_bin)
            self.ub_bin = np.array(ub_bin)
        else:
            self.lb_bin = lb_bin
            self.ub_bin = ub_bin

    required_artifacts = ["data", "sensitive_feature"]

    def _setup(self):
        self.pool_rankings = np.array(self.data.y.rankings)
        self.pool_sensitive_features = np.array(self.data.sensitive_feature)
        self.sf_name = self.data.sensitive_feature.name
        if self.k is None:
            self.k = int(len(self.pool_rankings) / 2)

        if self.down_sampling_step is None:
            self.down_sampling_step = max(int(len(self.pool_rankings) / 100), 1)

        # Sort ascending in parallel in case not already sorted
        p = self.pool_rankings.argsort()
        self.pool_rankings = self.pool_rankings[p]
        self.pool_sensitive_features = self.pool_sensitive_features[p]

        self.pool_groups = list(set(self.pool_sensitive_features))
        self.pool_items = np.arange(0, len(self.pool_rankings))
        self.num_items = len(self.pool_rankings)

        self.subset_sensitive_features = self.pool_sensitive_features[: self.k]
        self.subset_items = self.pool_items[: self.k]
        self.subset_groups = list(set(self.subset_sensitive_features))

        if "scores" in self.data.y:
            self.pool_scores = np.array(self.data.y.scores)
            if not np.issubdtype(self.pool_scores.dtype, np.number):
                raise ValidationError(
                    "`scores` array provided contains non-numeric elements."
                )

            self.subset_scores = self.pool_scores[: self.k]
        else:
            self.pool_scores = None
            self.subset_scores = None

        # if desired proportions are not provided, set it to the pool proportions
        if not all(self.desired_proportions):
            uniques, counts = np.unique(
                self.pool_sensitive_features, return_counts=True
            )
            self.desired_proportions = dict(zip(uniques, counts / self.num_items))

        return self

    def _validate_arguments(self):
        check_data_instance(self.data, TabularData)
        check_existence(self.data.sensitive_features, "sensitive_features")
        check_feature_presence("rankings", self.data.y, "y")
        check_artifact_for_nulls(self.data, "Data")

        return self

    def evaluate(self):
        """Runs the assessment process

        Returns
        -------
        dict, nested
            Key: assessment category
            Values: detailed results associated with each category
        """

        skew_results = self._skew()
        ndkl_results = self._ndkl()
        fins_results = self._fins()

        res = {**skew_results, **ndkl_results, **fins_results}
        res = {k: v for k, v in res.items() if k in METRIC_SUBSET}

        # Reformat results
        res = [pd.DataFrame(v).assign(metric_type=k) for k, v in res.items()]
        res = pd.concat(res)
        res[["type", "subtype"]] = res.metric_type.str.split("-", expand=True)
        res.drop("metric_type", axis=1, inplace=True)

        self.results = [MetricContainer(res, **self.get_container_info())]

        if self.pool_scores is not None:
            self._score_distribution()

        return self

    def _skew(self):
        """Claculates skew parity

        For every group, skew is the proportion of the selected candidates
            from that group over the desired proportion for that group.

        Returns
        -------
        dict
            skew parity difference
        """
        uniques, counts = np.unique(self.subset_sensitive_features, return_counts=True)
        subset_proportions = dict(zip(uniques, counts / self.k))

        skew = {}
        for g in self.pool_groups:
            sk = (subset_proportions[g] + EPSILON) / (
                self.desired_proportions[g] + EPSILON
            )
            skew[g] = sk

        skew = {
            "skew_parity_difference-score": [
                {"value": max(skew.values()) - min(skew.values())}
            ]
        }

        return skew

    def _kld(self, dist_1, dist_2):
        """Calculates KL divergence

        Parameters
        ----------
        dist_1 : list
            first distribution
        dist_2 : list
            second distribution

        Returns
        -------
        float
            KL divergence
        """
        vals = []
        for p1, p2 in zip(dist_1, dist_2):
            vals.append(p1 * math.log((p1 + EPSILON) / (p2 + EPSILON)))

        return sum(vals)

    def _ndkl(self):
        """Calculates normalized discounted cumulative KL-divergence (ndkl)

        It is based on the following paper:
            Geyik, Sahin Cem, Stuart Ambler, and Krishnaram Kenthapadi. "Fairness-aware ranking in search &
            recommendation systems with application to linkedin talent search."
            Proceedings of the 25th acm sigkdd international conference on knowledge discovery & data mining. 2019.

        Returns
        -------
        dict
            normalized discounted cumulative KL-divergence (ndkl)
        """
        Z = np.sum(1 / (np.log2(np.arange(1, self.num_items + 1) + 1)))

        total = 0.0
        for k in range(1, self.num_items + 1):
            item_attr_k = list(self.pool_sensitive_features[:k])
            item_distr = [
                item_attr_k.count(attr) / len(item_attr_k)
                for attr in self.desired_proportions.keys()
            ]
            total += (1 / math.log2(k + 1)) * self._kld(
                item_distr, list(self.desired_proportions.values())
            )

        ndkl = {"ndkl-score": [{"value": (1 / Z) * total}]}

        return ndkl

    def _fins(self):
        """Calculates group fairness metrics for subset selections from FINS paper and library

        It is based on the following paper:
            Cachel, Kathleen, and Elke Rundensteiner. "FINS Auditing Framework: Group Fairness for Subset Selections."
            Proceedings of the 2022 AAAI/ACM Conference on AI, Ethics, and Society. 2022.

        Returns
        -------
        dict
            fianress metrics
        """
        fins_metrics = {}

        # represent sensitive feature values via consecutive integers
        lookupTable, pool_sf_int = np.unique(
            self.pool_sensitive_features, return_inverse=True
        )
        lookupTable, subset_sf_int = np.unique(
            self.subset_sensitive_features, return_inverse=True
        )

        selectRt, parity_score = fins.parity(
            self.pool_items, pool_sf_int, self.subset_items, subset_sf_int
        )
        fins_metrics["demographic_parity_ratio-score"] = [{"value": parity_score}]

        propOfS, balance_score = fins.balance(
            pool_sf_int, self.subset_items, subset_sf_int
        )
        fins_metrics["balance_ratio-score"] = [{"value": balance_score}]

        # Score-dependant metrics
        if self.subset_scores is not None:
            AvgScore, score_parity_score = fins.score_parity(
                self.subset_items, self.subset_scores, subset_sf_int
            )
            fins_metrics["score_parity_ratio-score"] = [{"value": score_parity_score}]

            TotalScore, score_balance_score = fins.score_balance(
                self.subset_items, self.subset_scores, subset_sf_int
            )
            fins_metrics["score_balance_ratio-score"] = [{"value": score_balance_score}]

            if self.pool_scores is not None:
                RselectRt, relevance_parity_score = fins.relevance_parity(
                    self.pool_items,
                    self.pool_scores,
                    pool_sf_int,
                    self.subset_items,
                    self.subset_scores,
                    subset_sf_int,
                )
                fins_metrics["relevance_parity_ratio-score"] = [
                    {"value": relevance_parity_score}
                ]

                if self.q:
                    QselectRt, qualififed_parity_score = fins.qualififed_parity(
                        self.pool_items,
                        self.pool_scores,
                        pool_sf_int,
                        self.subset_items,
                        self.subset_scores,
                        subset_sf_int,
                        self.q,
                    )
                    fins_metrics["qualified_demographic_parity_ratio-score"] = [
                        {"value": qualififed_parity_score}
                    ]

                    QpropOfS, qualififed_balance_score = fins.qualified_balance(
                        self.pool_items,
                        self.pool_scores,
                        pool_sf_int,
                        self.subset_items,
                        self.subset_scores,
                        subset_sf_int,
                        self.q,
                    )
                    fins_metrics["qualified_balance_ratio-score"] = [
                        {"value": qualififed_balance_score}
                    ]

                if self.lb_bin is not None and self.ub_bin is not None:
                    (
                        bin_group_selection_proportions,
                        calibrated_parity_score,
                    ) = fins.calibrated_parity(
                        self.pool_items,
                        self.pool_scores,
                        pool_sf_int,
                        self.subset_items,
                        self.subset_scores,
                        subset_sf_int,
                        self.lb_bin,
                        self.ub_bin,
                    )
                    fins_metrics["calibrated_demographic_parity_ratio-score"] = [
                        {"value": calibrated_parity_score}
                    ]

                    (
                        bin_group_proportions,
                        calibrated_balance_score,
                    ) = fins.calibrated_balance(
                        self.pool_items,
                        self.pool_scores,
                        pool_sf_int,
                        self.subset_items,
                        self.subset_scores,
                        subset_sf_int,
                        self.lb_bin,
                        self.ub_bin,
                    )
                    fins_metrics["calibrated_balance_ratio-score"] = [
                        {"value": calibrated_balance_score}
                    ]

        return fins_metrics

    def _score_distribution(self):
        """Calculates scores empirical distribution curve for each demographic group"""

        groups = np.unique(self.pool_sensitive_features)
        for group in groups:
            ind = np.where(self.pool_sensitive_features == group)
            group_scores = self.pool_scores[ind]
            emp_dist_df = empirical_distribution_curve(
                group_scores, self.down_sampling_step, variable_name="scores"
            )
            emp_dist_df.name = "score_empirical_distribution"

            labels = {"sensitive_feature": self.sf_name, "group": group}

            e = TableContainer(
                emp_dist_df,
                **self.get_container_info(labels=labels),
            )
            self._results.append(e)
