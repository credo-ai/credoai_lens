"""Performs comparison between 2 pipelines"""
from typing import List, Literal, Optional
from credoai.evaluators.utils.validation import check_instance
from credoai.evidence.containers import MetricContainer
from credoai.lens import Lens

from credoai.utils import flatten_list
from credoai.prism.comparators.metric_comparator import MetricComparator


class Compare:
    SUPPORTED_CONTAINERS = [MetricContainer]

    def __init__(
        self,
        pipelines: List[Lens],
        ref: Optional[str] = None,
        operation: Literal["diff", "ratio"] = "diff",
        abs: bool = False,
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
        self.pipelines = pipelines
        self.ref = ref
        self.operation = operation
        self.abs = abs
        self._validate()
        self._extract_results_containers()

    def _validate(self):
        for evaluator in self.pipelines:
            check_instance(evaluator, Lens)

    def _extract_results_containers(self):
        self.containers = flatten_list([x.pipeline for x in self.pipelines])
        self.containers = flatten_list([x.evaluator.results for x in self.containers])
        # Remove unsupported containers
        self.supported_results = [
            x for x in self.containers if type(x) in self.SUPPORTED_CONTAINERS
        ]

    def run(self):
        # TODO: Add a splitter for different type of containers
        # When we have other comparators we can use the suitable ones depending
        # on container type. Potentially also dependent on evaluator
        self.results = MetricComparator(
            self.supported_results, self.ref, self.operation, self.abs
        ).compare()
