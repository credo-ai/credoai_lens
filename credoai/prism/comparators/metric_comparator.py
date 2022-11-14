"""Comparators for metric type containers"""
from typing import Callable, Literal, Optional

import numpy as np
from pandas import DataFrame, concat

from credoai.evaluators.utils.validation import check_instance
from credoai.evidence import MetricContainer
from credoai.prism.comparators.comparator import Comparator
from credoai.utils.common import ValidationError


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

    OPERATIONS = {
        "diff": lambda x, ref: x - ref,
        "ratio": lambda x, ref: x / ref,
        "perc": lambda x, ref: x * 100 / ref,
        "perc_diff": lambda x, ref: ((x - ref) / ref) * 100,
    }

    def __init__(
        self,
        EvidenceContainers,
        ref: Optional[str] = None,
        operation: Literal["diff", "ratio"] = "diff",
        abs: bool = False,
    ):
        # attributes all comparators will need
        self.overall_ref = ref
        self.operation = self.OPERATIONS[operation]
        self.abs = abs
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

        if self.overall_ref not in self.EvidenceContainers.keys():
            raise ValidationError("Reference ID not found.")

    def compare(self):
        """
        Runs all comparisons
        """
        # Calculate scalar differences across all the metrics
        self._scalar_operation()
        # Calculate overall stats for a metric result
        self._run_superlative()
        return self

    def _scalar_operation(self) -> DataFrame:
        """
        Compares all each metric to a specific reference value

        Returns
        -------
        DataFrame
            Columns:

                type: metric name
                value: original value of the metric
                id: Identifier of the origin of the metric
                comparison: value of the comparison
        """
        all_res = [
            res.df.assign(id=ide) for ide, res in self.EvidenceContainers.items()
        ]
        all_res = concat(all_res, ignore_index=True)
        # Group by metrics and calculate comparison
        output = []
        for _, results in all_res.groupby("type"):
            ref_value = results.value.loc[results.id == self.overall_ref].iloc[0]
            if ref_value:
                results["comparison"] = self.operation(results.value, ref_value)
            else:
                results["comparison"] = None
            output.append(results)
        output = concat(output, ignore_index=True)

        if self.abs:
            output["comparison"] = output.comparison.abs()
        self.comparisons["scalar_comparison"] = output

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
