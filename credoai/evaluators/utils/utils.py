from credoai.evaluators import Evaluator


def string2evaluator(str_in):
    for eval in Evaluator.__subclasses__():
        if str_in == eval.__name__:
            return eval
    raise Exception(
        f"<{str_in}> not found in list of Evaluators. Please confirm specified evaluator name is identical to Evaluator class definition."
    )
