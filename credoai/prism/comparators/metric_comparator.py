"""Comparators for metric type containers"""
from typing import Callable
from credoai.prism import Comparator
from credoai.evaluators.utils.validation import check_instance
from credoai.evidence import MetricContainer
from credoai.utils.common import ValidationError

import pandas as pd
import numpy as np
from copy import deepcopy


class MetricComparator(Comparator):
    """
    Class for comparing metric evidence objects

    Supported comparisons for Metrics include difference, maximal values for each metric,
    and minimal values for each metric. Comparisons are evaluated on intersection of the
    metrics represented in the two provided MetricContainer objects. Comparison values for
    non-intersecting metrics will be NoneType wrapped in output container.

    Inputs:
        EvidenceContainers: dictionary of {name_of_model: MetricContainer} key-value pairs

    Output, stored in a LensComparison object, is result of comparisons.
    """

    def __init__(self, EvidenceContainers):
        # attributes all comparators will need
        super().__init__(EvidenceContainers)

    def _setup(self):
        """Extracts all the results from the result containers"""
        for container in self.EvidenceContainers.values():
            for metric in container.df.type:
                self.evaluations.add(metric)

    def _validate(self):
        """
        Check that provided containers are all MetricContainer type
        Check that len >= 2
        """
        for container in self.EvidenceContainers.values():
            check_instance(container, MetricContainer)

        if len(self.EvidenceContainers) < 2:
            raise ValidationError("Expected multiple evidence objects to compare.")

    def compare(self):
        """
        Runs all comparisons
        """
        # Calculate scalar differences across all the metrics
        self._scalar_difference()
        # Calculate overall stats for a metric result
        self._run_superlative()
        return self

    def _scalar_difference(self, abs=False):
        """
        Calculates the scalar difference across all results for a specific metric.

        Parameters
        ----------
        abs : bool, optional
            If true calculates absolute difference, by default False

        Returns
        -------
        Adds an output dictionary to self.comparisons.
        Dict structure:

            Keys: metric names
            Values: pd.DataFrame objects, each with shape len(self.EvidenceContainers), len(self.EvidenceContainers)

        DataFrame i contains results for metric i:

            Pairwise difference between MetricContainers j and k
            If abs == True, return the absolute difference between metrics results
            If metric is not measured for Container j or k, DataFrame[j, k] is None

        """

        self.comparisons["scalar_difference"] = {}
        for metric in self.evaluations:
            comparison_dict = {}
            for model_1, metric_ev1 in self.EvidenceContainers.items():
                comparisons = []
                if metric not in metric_ev1.df.type.values:
                    # Needing to assume the underlying dataframe will have column "type"
                    # seems a little brittle
                    comparisons = [None] * len(self.EvidenceContainers)
                else:
                    for metric_ev2 in self.EvidenceContainers.values():
                        if metric in metric_ev2.df.type.values:
                            # Assuming the column with the value will be named "value" is
                            # brittle. As is needing to check the 0-index to get the metric result
                            comparisons.append(
                                metric_ev1.df[
                                    metric_ev1.df["type"] == metric
                                ].value.iloc[0]
                                - metric_ev2.df[
                                    metric_ev2.df["type"] == metric
                                ].value.iloc[0]
                            )
                            if abs:
                                comparisons[-1] = np.abs(comparisons[-1])
                        else:
                            comparisons.append(None)

                comparison_dict[model_1] = comparisons
            comparison_df = pd.DataFrame.from_dict(
                comparison_dict, orient="index", columns=self.EvidenceContainers.keys()
            )
            self.comparisons["scalar_difference"][metric] = comparison_df

    def _superlative_eval(self, superlative: Callable, superlative_name: str):
        """
        Calculate maximals in results distribution.

        Parameters
        ----------
        superlative : Callable
            Function to apply to the results metrics
        superlative_name : str
            Identifier of the operation applied to the list.
        """
        self.comparisons[superlative_name] = {}
        for metric in self.evaluations:
            self.comparisons[superlative_name][metric] = superlative(
                [
                    ev.df[ev.df["type"] == metric].value.iloc[0]
                    for ev in self.EvidenceContainers.values()
                ]
            )

    def _run_superlative(self):
        """
        Runs calculation for min, max, and mean values for each self.evaluation

        Useful to find overall summary stats metrics; E.g. want to get upper bound on parity gaps across models

        """
        eval_to_run = {"highest": max, "lowest": min, "mean": np.mean}

        for eval_name, eval in eval_to_run.items():
            self._superlative_eval(eval, eval_name)
