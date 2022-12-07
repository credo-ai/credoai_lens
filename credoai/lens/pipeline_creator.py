"""
Classes responsible for programatically creating pipelines of evaluators
"""

import inspect
from collections import defaultdict
from typing import List

import credoai.evaluators
from connect.evidence import EvidenceRequirement
from connect.governance import Governance
from credoai.evaluators import *
from credoai.evaluators.utils import name2evaluator
from credoai.utils.common import remove_suffix


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
    kwargs: dict = defaultdict(dict)
    for e in evidence_requirements:
        labels = e.label
        evaluator_name = labels.get("evaluator")
        if evaluator_name is None:
            continue
        evaluators.add(evaluator_name)
        if evaluator_name in ["ModelFairness", "Performance"]:
            if "metrics" not in kwargs[evaluator_name]:
                kwargs[evaluator_name]["metrics"] = extract_metrics(labels)
            else:
                kwargs[evaluator_name]["metrics"] |= extract_metrics(labels)
        if evaluator_name == "FeatureDrift":
            if "table_name" in labels:
                if labels["table_name"] == "Characteristic Stability Index":
                    kwargs[evaluator_name]["csi_calculation"] = True

    pipeline = []
    for evaluator_name in evaluators:
        evaltr_class = name2evaluator(evaluator_name)
        evaltr_kwargs = kwargs.get(evaluator_name, {})
        initialized_evaltr = evaltr_class(**evaltr_kwargs)
        pipeline.append(initialized_evaltr)
    return pipeline


def extract_metrics(labels):
    """Extract metrics from a single evidence requirement"""
    metrics = set()
    if "metric_type" in labels:
        metrics.add(remove_suffix(labels["metric_type"], "_parity"))
    elif "metric_types" in labels:
        metrics = metrics.union(labels["metric_types"])
    return metrics


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
