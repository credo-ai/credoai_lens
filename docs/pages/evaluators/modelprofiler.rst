
Model Profiler
==============


Model profiling evaluator.

This evaluator builds a model card the purpose of which is to characterize
a fitted model.

The overall strategy is:
    1. Extract all potentially useful info from the model itself in an
        automatic fashion.
    2. Allow the user to personalize the model card freely.

The method generate_template() provides a dictionary with several entries the
user could be interested in filling up.

Parameters
----------
model_info : Optional[dict]
    Information provided by the user that cannot be inferred by
    the model itself. The dictionary con contain any number of elements,
    a template can be provided by running the generate_template() method.

    The only restrictions are checked in a validation step:
    1. Some keys are protected because they are used internally
    2. Only basic python types are accepted as values

