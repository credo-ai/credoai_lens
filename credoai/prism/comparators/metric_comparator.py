"""Comparators for metric type containers"""
from typing import Callable, List, Literal, Optional

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
        EvidenceContainers: List[MetricContainer],
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
        self._extract_results_from_containers()

    def _validate(self):
        """
        Check that provided containers are all MetricContainer type
        Check that len >= 2
        """
        for container in self.EvidenceContainers:
            check_instance(container, MetricContainer)

        if len(self.EvidenceContainers) < 2:
            raise ValidationError("Expected multiple evidence objects to compare.")

    def compare(self):
        """
        Runs all comparisons
        """
        # Calculate scalar differences across all the metrics
        self._scalar_operation()
        return self

    def _extract_results_from_containers(self):
        # Assign model name as ID
        self.all_results = [
            res.df.assign(id=res.metadata["model_name"])
            for res in self.EvidenceContainers
        ]
        self.all_results = concat(self.all_results, ignore_index=True)

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
        # Group by metrics and calculate comparison
        output = []
        for _, results in self.all_results.groupby("type"):
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
