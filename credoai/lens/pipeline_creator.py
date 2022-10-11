import inspect
from collections import defaultdict
from typing import List

import credoai.evaluators
from credoai.evaluators import *
from credoai.evidence import EvidenceRequirement
from credoai.governance import Governance


class PipelineCreator:
    @staticmethod
    def generate_all_evaluators():
        all_evaluators = build_list_of_evaluators()
        return all_evaluators

    @staticmethod
    def generate_from_governance(governance: Governance):
        evidence_requirements = governance.get_evidence_requirements()
        governance_pipeline = process_evidence_requirements(evidence_requirements)
        return governance_pipeline


def process_evidence_requirements(evidence_requirements: List[EvidenceRequirement]):
    evaluators = set()
    kwargs = defaultdict(dict)
    for e in evidence_requirements:
        labels = e.label
        evaluator = labels.get("evaluator")
        if evaluator is None:
            continue
        evaluators.add(evaluator)
        # Ugly, must change in the future! If it needs to be hardcoded per evaluator
        # should make that part of evaluator class, or some helper function
        if evaluator in ["ModelFairness", "Performance"]:
            metrics = kwargs[evaluator].get("metrics", set())
            if "metric_type" in labels:
                metrics.add(labels["metric_type"])
            elif "metric_types" in labels:
                metrics = metrics.union(labels["metric_types"])
            kwargs[evaluator]["metrics"] = metrics

    pipeline = []
    for evaltr in evaluators:
        evaltr_class = eval(evaltr)
        evaltr_kwargs = kwargs.get(evaltr, {})
        initialized_evaltr = evaltr_class(**evaltr_kwargs)
        pipeline.append(initialized_evaltr)
    return pipeline


def build_list_of_evaluators():
    """
    Takes all the evaluator type objects available in Lens package
    and converts them to a list of instantiated objects. Only
    uses default values.

    Returns
    -------
    List(Evaluator types)
        List of instantiated evaluators
    """
    all_evaluators = []
    for x in dir(credoai.evaluators):
        try:
            evaluated = eval(x)
            if (
                inspect.isclass(evaluated)
                and issubclass(evaluated, Evaluator)
                and not inspect.isabstract(evaluated)
            ):
                all_evaluators.append(evaluated)
        except NameError:
            pass
    return [x() for x in all_evaluators]
