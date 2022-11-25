"""Performs comparison between 2 pipelines"""
from typing import List, Literal, Optional
from credoai.evaluators.utils.validation import check_instance
from credoai.evidence.containers import MetricContainer
from credoai.lens import Lens

from credoai.utils import flatten_list
from credoai.prism.comparators.metric_comparator import MetricComparator
from credoai.utils.common import ValidationError
from credoai.prism.task import Task


class Compare(Task):
    """
    Compare results across multiple pipelines.

    This class coordinates prism.comparators objects in order to execute
    comparisons across an arbitrary amount of Lens objects.

    The code execute the following steps:

    1. Extract results from the Lens objects
    2. Call the suitable comparators depending on the results container type
    3. Aggregate results from multiple comparators (currently only metrics supported)

    Parameters
    ----------
    pipelines : List[Lens]
        List of Lens objects, this must have been previously run, results need to be available
    ref : Optional[str], optional
        The model/dataset name by which to compare all others. Model/dataset names are
        defined when instantiating Lens objects, by the usage of credo.artifacts. If None, the
        first in the list will be used as a reference, by default None.
    operation : Literal["diff", "ratio", "perc", "perc_diff"], optional
        Indicates which operation is computed during the comparison. The accepted
        options are:

            "diff": x - ref,
            "ratio": x / ref,
            "perc": x * 100 / ref,
            "perc_diff": ((x - ref) / ref) * 100,

    abs : bool, optional
        If true the absolute value of the operation is returned, by default False
    """

    SUPPORTED_CONTAINERS = [MetricContainer]

    def __init__(
        self,
        ref_type="model",
        ref: Optional[str] = None,
        operation: Literal["diff", "ratio", "perc", "perc_diff"] = "diff",
        abs: bool = False,
    ):
        self.ref = ref
        self.ref_type = ref_type
        self.operation = operation
        self.abs = abs
        super().__init__()

    def _validate(self):
        for evaluator in self.pipelines:
            check_instance(evaluator, Lens)
        if len(self.pipelines) < 2:
            raise ValidationError("At least 2 lens objects are needed for a comparison")
        if not self.ref:
            self.ref = self.pipelines[0].__dict__[self.ref_type].name
            # TODO: LOG this -> (f"Reference {self.ref_type}: {self.ref}")

    def _setup(self):
        pipesteps = flatten_list([x.pipeline for x in self.pipelines])
        # Propagate step identifier to results
        for step in pipesteps:
            for result in step.evaluator.results:
                result.id = step.identifier
        self.containers = flatten_list([x.evaluator.results for x in pipesteps])
        # Remove unsupported containers
        self.supported_results = [
            x for x in self.containers if type(x) in self.SUPPORTED_CONTAINERS
        ]

    def run(self):
        # TODO: Add a splitter for different type of containers
        # When we have other comparators we can use the suitable ones depending
        # on container type. Potentially also dependent on evaluator
        self.results = MetricComparator(
            self.supported_results, self.ref_type, self.ref, self.operation, self.abs
        ).compare()
        return self

    def get_results(self):
        return self.results.comparisons
