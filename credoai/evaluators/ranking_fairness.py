from collections import Counter

import math
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
from credoai.utils.common import NotRunError
from credoai.evidence.containers import MetricContainer

EPSILON = 1e-12


class RankingFairness(Evaluator):
    """Ranking fairness evaluator for Credo AI

    This module takes in ranking results and
        provides functionality to perform fairness assessment

    Parameters
    ----------
    rankings : pandas.Series
        The computed ranks
    sensitive_features : pandas.Series
        A series of the sensitive feature labels (e.g., "male", "female") which should be used to create subgroups
    desired_proportions : dict, Optional
        The desired proportion for each subgroups (e.g., {"male":0.4, "female":0.6})
        If not provided, equal proportions are used
    skew_log : bool, Optional
        If True, will return logarithm of skew values.
    """

    def __init__(self, desired_proportions: dict = None, skew_log: bool = False):
        self.desired_proportions = (desired_proportions,)
        self.skew_log = skew_log

    name = "RankingFairness"
    required_artifacts = ["data", "sensitive_feature"]

    def _setup(self):
        self.sensitive_features = np.array(self.data.sensitive_feature.iloc[:, 0])
        self.rankings = np.array(self.data.y.iloc[:, 0])
        self.groups = list(set(self.sensitive_features))

        # Sort ascending in parallel in case not already sorted
        p = self.rankings.argsort()
        self.rankings = self.rankings[p]
        self.sensitive_features = self.sensitive_features[p]

        if not all(self.desired_proportions):
            proportions = [1.0 / len(self.groups)] * len(self.groups)
            self.desired_proportions = dict(zip(self.groups, proportions))

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
        self._results = {}

        skew_results = self._skew()
        ndkl_results = self._ndkl()

        self._results.update(
            {
                **skew_results,
                **ndkl_results,
            }
        )
        self.results = self._prepare_results()
        return self

    def _prepare_results(self):
        """Prepares results for export to Credo AI's Governance Platform

        Structures a subset of results for export as a dataframe with appropriate structure
        for exporting. See credoai.modules.credo_module.

        Returns
        -------
        pd.DataFrame
        Raises
        ------
        NotRunError
            If results have not been run, raise
        """
        if self._results is not None:
            metric_types = [
                "minimum_skew-score",
                "maximum_skew-score",
                "NDKL-score",
            ]
            prepared_arr = []
            index = []
            for metric_type in metric_types:
                if metric_type not in self.results:
                    continue
                val = self.results[metric_type]
                # if multiple values were calculated for metric_type
                # add them all. Assumes each element of list is a dictionary with a "value" key,
                # and other optional keys as metricmetadata
                if isinstance(val, list):
                    for l in val:
                        index.append(metric_type)
                        prepared_arr.append(l)
                else:
                    # assumes the dictionary has a "value" key, along with other optional keys
                    # as metric metadata
                    if isinstance(val, dict):
                        tmp = val
                    elif isinstance(val, (int, float)):
                        tmp = {"value": val}
                    index.append(metric_type)
                    prepared_arr.append(tmp)
            res = pd.DataFrame(prepared_arr, index=index).rename_axis(
                index="metric_type"
            )
            res = res.reset_index()
            res[["type", "subtype"]] = res.metric_type.str.split("-", expand=True)
            res.drop("metric_type", axis=1, inplace=True)
            return [MetricContainer(res)]
        else:
            raise NotRunError("Results not created yet. Call 'run' to create results")

    def _skew(self):
        """Claculates skew metrics

        For every group, skew is the proportion of the selected candidates
            from that group over the desired proportion for that group.
            Skew is sensitive to the number of top candidates included (k).
            Ideal skew is 100%.

        Returns
        -------
        dict
            min_skew: signifies the worst disadvantage in representation given to candidates
                from a specific group. Ideal value is 1.
            max_skew: signifies the largest unfair advantage provided to candidates from
                a specific group. Ideal value is 1.
        """
        actual_counts = dict(Counter(self.sensitive_features))
        actual_proportions = {
            k: v / len(self.sensitive_features) for k, v in actual_counts.items()
        }
        skew = {}
        for g in self.groups:
            sk = (actual_proportions[g] + EPSILON) / (
                self.desired_proportions[g] + EPSILON
            )
            if self.skew_log:
                sk = math.log(sk)
            skew[g] = sk

        skew_extrema = {
            "minimum_skew-score": min(skew.values()),
            "maximum_skew-score": max(skew.values()),
        }

        return skew_extrema

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
        """Calculates normalized discounted cumulative KL-divergence (NDKL)

        NDKL is bias measure that accounts for ranks. It is non-negative, with larger values
            indicating a greater divergence between the desired and actual distributions of
            sensitive attribute labels. NDKL equals 0 in the ideal fairness case of the two
            distributions being identical.

        Returns
        -------
        dict
            NDKL: normalized discounted cumulative KL-divergence.
                Ideal value is 0.
        """
        n_items = len(self.sensitive_features)
        Z = np.sum(1 / (np.log2(np.arange(1, n_items + 1) + 1)))

        total = 0.0
        for k in range(1, n_items + 1):
            item_attr_k = list(self.sensitive_features[:k])
            item_distr = [
                item_attr_k.count(attr) / len(item_attr_k)
                for attr in self.desired_proportions.keys()
            ]
            total += (1 / math.log2(k + 1)) * self._kld(
                item_distr, list(self.desired_proportions.values())
            )

        ndkl = {"NDKL-score": (1 / Z) * total}

        return ndkl
