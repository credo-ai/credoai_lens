"""Performs comparison between 2 pipelines"""
from typing import List
from comparators.metric_comparator import MetricComparator
from credoai.evidence.containers import EvidenceContainer, MetricContainer


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
        """Remove the unsupported type of containers."""
        self.results_primary = [
            x for x in self.results_primary if type(x) in self.supported_containers
        ]
        self.results_secondary = [
            x for x in self.results_secondary if type(x) in self.supported_containers
        ]
        # Assumption on the shape of the list of containers.
        # TODO: relax the assumptions in future iterations
        # 1.

    def _form_pairs(self):
        print("stuff")
        pass
