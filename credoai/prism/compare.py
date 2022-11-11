"""Performs comparison between 2 pipelines"""
from typing import List
from comparators.metric_comparator import MetricComparator
from credoai.evidence.containers import EvidenceContainer, MetricContainer

from credoai.utils import ValidationError


class ComparePairs:
    def __init__(
        self,
        results_primary: List[EvidenceContainer],
        results_secondary: List[EvidenceContainer],
    ):
        """
        Compare results between 2 different Lens runs.

        Depending on the type of containers detected, this class will invoke
        the appropriate comparator.

        Parameters
        ----------
        results_primary : List[EvidenceContainer]
            List of results from a Lens run, in the cases in which a reference result
            is needed, this is considered such.
        results_secondary : List[EvidenceContainer]
            List of results from a Lens run
        """
        self.results_primary = results_primary
        self.results_secondary = results_secondary
        self.supported_containers = [MetricContainer]
        self.primary_evaluators = self
        self._validate()

    def _validate(self):
        # Assumption on the shape of the list of containers.
        # TODO: relax the assumptions in future iterations
        # 1. They are the same length
        if len(self.results_primary) != len(self.results_secondary):
            raise ValidationError("List of results have different length")
        # 2. They have the same type of containers
        if [type(x) for x in self.results_primary] != [
            type(x) for x in self.results_secondary
        ]:
            raise ValidationError(
                "Containers type are different across the lists of results"
            )

        """Remove the unsupported type of containers."""
        self.results_primary = [
            x for x in self.results_primary if type(x) in self.supported_containers
        ]
        self.results_secondary = [
            x for x in self.results_secondary if type(x) in self.supported_containers
        ]

    def _create_pairs(self):
        """Create a list of results pair"""
        print("stuff")
        pass
