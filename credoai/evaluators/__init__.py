"""
Perform specific evaluations on model and/or dataset
"""
from .utils import list_evaluators

usable_evaluators = list_evaluators()
globals().update({k: mod for k, mod in usable_evaluators})
