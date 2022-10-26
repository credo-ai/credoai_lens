from credoai.evaluators import Evaluator

from credoai.utils.common import ValidationError
from credoai.modules.deepchecks_constants import DEFAULT_CHECKS


class Deepchecks(Evaluator):
    """
    deepchecks evaluator

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
    checks : List-like, optional
        A list of instantiated deepchecks checks objects (e.g. BoostingOverfit, CalibrationScore)
        #TODO allow list of strings?
    """

    required_artifacts = {"model", "data"}

    # TODO always pass core dataframe, note deepchecks data object
    # Guess we always pass both or just test -- it will do single data set, I guess

    def __init__(self, checks=DEFAULT_CHECKS):
        super().__init__()
        self.checks = checks

    def _setup(self):
        # Set artifacts
        pass

    def _setup_deepchecks(self):
        # validate deepchecks objects
        # construct Suite
        pass

    def evaluate(self):
        """
        Execute any data/model processing required for the evaluator.

        Populates the self.results object.

        Returns
        -------
        self
        """
        # run suite
        # convert suite results to json and package into deepchecks evidence
        return self

    def _validate_arguments(self):
        """
        Check that basic requirements for the run of an evaluator are met.
        """
        pass
