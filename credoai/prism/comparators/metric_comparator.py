"""Comparators for metric type containers"""
from typing import List

from connect.evidence import MetricContainer
from pandas import DataFrame, concat

from credoai.evaluators.utils.validation import check_instance
from credoai.prism.comparators.comparator import Comparator


class MetricComparator(Comparator):
    """
    Class for comparing metric evidence objects.

    Each metric is compared to the respective reference value. The reference value is the metric value
    associated to a specific model or dataset. The reference model/dataset are identified by the user, see
    ref type and ref in Parameters.

    Supported comparisons for Metrics include differences, ratio, percentage ratio, and percentage
    difference.

    Parameters
    ----------
    EvidenceContainers : List[MetricContainer]
        A list of metric containers.
    ref_type: str
        Accepted values: model, assessment_data, training_data. Indicates which of the
        artifacts should be used as a refence, by default model.
    ref : str
        The model/dataset name by which to compare all others. Model/dataset names are
        defined when instantiating Lens objects, by the usage of credo.artifacts.
    operation : str
        Accepted operations: "diff", "ratio", "perc", "perc_diff", by default "diff"
    abs : bool, optional
        If true the absolute value of the operation is returned, by default False
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
        operation: str = "diff",
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
        """
        for container in self.EvidenceContainers:
            check_instance(container, MetricContainer)

    def compare(self):
        """
        Runs all comparisons
        """
        # Calculate scalar differences across all the metrics
        self.comparisons["scalar_comparison"] = self._scalar_operation()
        return self

    def _extract_results_from_containers(self):
        """
        Extract results from containers.
        """
        # Create id columns from the result id.
        self.all_results = [
            res.data.assign(id=res.id) for res in self.EvidenceContainers
        ]
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

        # Remove reference type from ids, and add metric type to create unique groups
        to_grp_by = [x for x in self.ID_COLS if x != self.ref_type] + ["type"]

        for _, results in self.all_results.groupby(to_grp_by):
            # Define reference value
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

        return final
