import inspect
from credoai.utils.common import dict_hash
from credoai.evaluators import *
import credoai.evaluators


import functools


def log_command(fun):
    @functools.wraps(fun)
    def wrapper(self, *args, **kwargs):
        tmp = fun(self, *args, **kwargs)
        name = fun.__name__
        self.command_list.append(get_command_string(name, args, kwargs))
        return tmp

    return wrapper


def get_command_string(name, arg, kwarg):
    arg_parse = [get_arg_info(arg) for arg in arg]
    kwarg_parse = [f"{k}={get_arg_info(v)}" for k, v in kwarg.items()]
    return f"{name}({','.join([x for x in arg_parse + kwarg_parse if x is not None])})"


def get_arg_info(arg):
    if callable(arg):
        # Get only initialization arguments
        init_args = {
            k: v
            for k, v in arg.__dict__.items()
            if k in list(inspect.signature(arg.__init__).parameters.keys())
        }
        return f"{type(arg).__name__}({get_string_of_arguments_from_kwargs(init_args)})"
    elif isinstance(arg, int):
        return arg
    elif isinstance(arg, str):
        return f'"{arg}"'


def get_string_of_arguments_from_kwargs(keyarg):
    return ",".join([f"{x[0]}={check_int_str(x[1])}" for x in keyarg.items()])


def check_int_str(x):
    if isinstance(x, (int, float)):
        return x
    elif isinstance(x, str):
        return f'"{x}"'


def add_metric_keys(prepared_results):
    """Adds metric keys to prepared results

    Metric keys are used to associated charts, html blobs, and other assets with
    specific metrics. They are a hash of most of the metric's attributes, except the value.
    So if a metric changes value, the key will stay the same.

    Metric keys should be defined after all pertinent information is appended to a metric.
    Lens normally handles key association, because it may add additional metadata to a metric
    beyond what the assessment creates (e.g., dataset name, model name, etc.)

    Parameters
    ----------
    prepared_results : DataFrame
        output of CredoAssessment.prepare_results()
    """
    if prepared_results is None:
        return
    ignored = ["value", "metadata"]
    keys = [
        dict_hash({k: v for k, v in metric_dict.items() if k not in ignored})
        for metric_dict in prepared_results.reset_index().to_dict("records")
    ]
    prepared_results["metric_key"] = keys


def build_list_of_evaluators():
    all_evaluators = []
    for x in dir(credoai.evaluators):
        try:
            if (
                inspect.isclass(eval(x))
                and issubclass(eval(x), Evaluator)
                and not inspect.isabstract(eval(x))
            ):
                all_evaluators.append(eval(x))
        except NameError:
            pass
    return [x() for x in all_evaluators]
