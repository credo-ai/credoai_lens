import functools
import inspect
from typing import Callable, Union

import credoai.evaluators
from credoai.utils.common import dict_hash


def log_command(fun: Callable):
    """
    Decorator loggin the full isgnature of a function call.

    Parameters
    ----------
    fun : Callable
        A generic function, specifically used for Lens.add, Lens.delete, Lens.run

    """

    @functools.wraps(fun)
    def wrapper(self, *args, **kwargs):
        tmp = fun(self, *args, **kwargs)
        name = fun.__name__
        self.command_list.append(get_command_string(name, args, kwargs))
        return tmp

    return wrapper


def get_command_string(name: str, arg: dict, kwarg: dict) -> str:
    """
    Combines name and function arguments into a signature string.

    Parameters
    ----------
    name : str
        Function's name.
    arg : dict
        Function's positional arguments.
    kwarg : dict
        Function's keyword argumwents

    Returns
    -------
    str
        Full function signature,e.g., fun_name(fun_arg1, fun_arg1..., fun_kwarg1...)
    """

    arg_parse = [get_arg_info(arg) for arg in arg]
    kwarg_parse = [f"{k}={get_arg_info(v)}" for k, v in kwarg.items()]
    all_arguments = arg_parse + kwarg_parse
    return f"{name}({','.join([x for x in all_arguments if x is not None])})"


def get_arg_info(arg: Union[Callable, str, int]) -> str:
    """
    Takes a function's arguments and converts them into strings.

    Parameters
    ----------
    arg : Union[Callable, str, int]
        This is quite custom made for usage in Lens(). The positional arguments
        can be a call to a class, or int/str. This handles all cases.

    Returns
    -------
    Union[str,int]
        Either a string representing the function signature, or str/int
        depending on the argument.
    """
    if callable(arg):
        # Get only initialization arguments
        init_args = {
            k: v
            for k, v in arg.__dict__.items()
            if k in list(inspect.signature(arg.__init__).parameters.keys())
        }
        return f"{type(arg).__name__}({get_string_of_arguments_from_kwargs(init_args)})"
    elif isinstance(arg, int):
        return str(arg)
    elif isinstance(arg, str):
        return f'"{arg}"'


def get_string_of_arguments_from_kwargs(keyarg: dict) -> str:
    """
    Transform positional arguments in string.

    Parameters
    ----------
    keyarg : dict
        Function's positional arguments.

    Returns
    -------
    str
        String representing the positional arguments
    """
    return ",".join([f"{x[0]}={check_int_str(x[1])}" for x in keyarg.items()])


def check_int_str(x: Union[int, float, str]) -> Union[int, str, float]:
    """
    Check what type is the argument and reformats in case it is a string.
    """
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
