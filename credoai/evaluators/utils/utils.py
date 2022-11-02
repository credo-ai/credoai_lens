from credoai.evaluators import Evaluator


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
