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
    ID_COLS = [
        "evaluator",
        "model",
        "assessment_data",
        "training_data",
        "sensitive_feature",
    ]

    def __init__(
        self,
        EvidenceContainers: List[MetricContainer],
        ref_type: str,
        ref: str,
        operation: Literal["diff", "ratio"] = "diff",
        abs: bool = False,
    ):
        # attributes all comparators will need
        self.ref_type = ref_type
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
        self.all_results = [res.df.assign(id=res.id) for res in self.EvidenceContainers]
        self.all_results = concat(self.all_results, ignore_index=True)
        self.all_results[self.ID_COLS] = self.all_results.id.str.split("~", expand=True)

        # Drop columns with no variance
        self.all_results.drop("id", axis=1, inplace=True)

    def _scalar_operation(self) -> DataFrame:
        """
        Compares all each metric to a specific reference value

        Returns
        -------
        DataFrame
            Containing a comparison column with the results. The other columns
            include initial values and all other necessary identifiers.
        """
        # Group by metrics and calculate comparison
        output = []

        # Remove reference id from ids, and add type to create unique groups
        to_grp_by = [x for x in self.ID_COLS if x != self.ref_type] + ["type"]

        for _, results in self.all_results.groupby(to_grp_by):

            ref_value = results.value.loc[
                results[self.ref_type] == self.overall_ref
            ].iloc[0]

            if ref_value:
                results["comparison"] = self.operation(results.value, ref_value)
            else:
                results["comparison"] = None
            output.append(results)

        final = concat(output, ignore_index=True)

        if self.abs:
            final["comparison"] = final["comparison"].abs()

        # Clean data frame
        nunique = final.nunique()
        cols_to_drop = nunique[nunique == 1].index
        final = final.drop(cols_to_drop, axis=1)
        final[final == "NA"] = None

        self.comparisons["scalar_comparison"] = final
