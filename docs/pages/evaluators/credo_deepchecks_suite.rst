
Credo_deepchecks_suite
======================


Deepchecks evaluator

This evaluator enables running of deepchecks `checks` and passing the results to
the Governance platform in the form of a deepchecks SuiteResult, cast to JSON format.
See https://docs.deepchecks.com/stable/api/generated/deepchecks.tabular.checks.model_evaluation.html
and https://docs.deepchecks.com/stable/api/generated/deepchecks.core.SuiteResult.html
and https://docs.deepchecks.com/stable/user-guide/general/customizations/examples/plot_create_a_custom_suite.html
for more details.

This evaluator provides some redundant functionality. For instance, metrics which can be
calculated using the Performance evaluator can potentially be calculated by deepchecks
(and thus this evaluator) as well. The same applies to the FeatureDrift evaluator.
When a choice exists, the best practice dictates that the "Lens native" evaluator should
be used in preference to deepchecks, since output formats of other evaluators is generally
consistent, while this deepchecks evaluator outputs results in a highly structured JSON format.


Parameters
----------
name : str, optional
    Name of the supplied deepchecks suite
checks : List-like, optional
    A list of instantiated deepchecks checks objects (e.g. BoostingOverfit, CalibrationScore)
    #TODO allow list of strings?
