import inspect
import os
import pkgutil

from credoai.evaluators.evaluator import Evaluator


def name2evaluator(evaluator_name):
    """Converts evaluator name to evaluator class"""
    for eval in all_subclasses(Evaluator):
        if evaluator_name == eval.__name__:
            return eval
    raise Exception(
        f"<{evaluator_name}> not found in list of Evaluators. Please confirm specified evaluator name is identical to Evaluator class definition."
    )


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


def list_evaluators_exhaustive():
    """List all defined evaluation modules"""
    evaluator_path = os.path.dirname(os.path.dirname(__file__))
    return pkgutil.iter_modules([evaluator_path])


def list_evaluators():
    """List subset of all defined evaluators where the module is importable"""
    evaluators = list(list_evaluators_exhaustive())
    importer = evaluators[0][0]
    # get loadable modules
    usable_modules = []
    for evaluator in evaluators:
        module = importer.find_module(evaluator[1])
        try:
            m = module.load_module()
            usable_modules.append((evaluator[1], m))
        except (ModuleNotFoundError, ImportError):
            pass
    # get evaluators from loadable modules
    usable_evaluators = []
    for module in usable_modules:
        try:
            for m in inspect.getmembers(module[1], inspect.isclass):
                if m[1].__module__ == module[0] and isinstance(m[1](), Evaluator):
                    usable_evaluators.append(m)
        except:
            pass
    return usable_evaluators
