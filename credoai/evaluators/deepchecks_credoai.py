from typing import List, Optional

from connect.evidence.deepchecks_evidence import DeepchecksContainer
from deepchecks.core import BaseCheck
from deepchecks.tabular import Dataset, Suite

from credoai.evaluators.evaluator import Evaluator
from credoai.evaluators.utils.validation import check_requirements_deepchecks
from credoai.modules.constants_deepchecks import DEFAULT_CHECKS


class Deepchecks(Evaluator):
    """
    `Deepchecks <https://docs.deepchecks.com/stable/getting-started/welcome.html?utm_campaign=/&utm_medium=referral&utm_source=deepchecks.com>`_ evaluator for Credo AI (Experimental)

    This evaluator enables running of deepchecks `checks` and passing the results to
    the Governance platform in the form of a deepchecks SuiteResult, cast to JSON format.
    See `model evaluation <https://docs.deepchecks.com/stable/api/generated/deepchecks.tabular.checks.model_evaluation.html>`_
    and `SuiteResults <https://docs.deepchecks.com/stable/api/generated/deepchecks.core.SuiteResult.html>`_
    and `create a custom suite <https://docs.deepchecks.com/stable/user-guide/general/customizations/examples/plot_create_a_custom_suite.html>`_
    for more details.

    This evaluator provides some redundant functionality. For instance, metrics which can be
    calculated using the Performance evaluator can potentially be calculated by deepchecks
    (and thus this evaluator) as well. The same applies to the FeatureDrift evaluator.
    When a choice exists, the best practice dictates that the "Lens native" evaluator should
    be used in preference to deepchecks, since output formats of other evaluators is generally
    consistent, while this deepchecks evaluator outputs results in a highly structured JSON format.

    Required Artifacts
    ------------------
        **Required Artifacts**

        Generally artifacts are passed directly to :class:`credoai.lens.Lens`, which
        handles evaluator setup. However, if you are using the evaluator directly, you
        will need to pass **at least one** of the following artifacts when instantiating the evaluator:

        - model: :class:`credoai.artifacts.Model` or :class:`credoai.artifacts.RegressionModel`
        - assessment_data: :class:`credoai.artifacts.TabularData`
            The assessment data to evaluate. Assessment data is used to calculate metrics
            on the model.
        - training_data: :class:`credoai.artifacts.TabularData`
            The training data to evaluate. The training data was used to tran the model


    Parameters
    ----------
    suite_name : str, optional
        Name of the supplied deepchecks suite
    checks : List[BaseCheck], optional
        A list of instantiated deepchecks checks objects (e.g. BoostingOverfit, CalibrationScore)
    """

    required_artifacts = {"model", "assessment_data", "training_data"}
    # all artifacts are OPTIONAL; All that's required is that at least one of these is
    # provided. The evaluator's custom validation function checks for this.

    def __init__(
        self,
        suite_name: Optional[str] = "Credo_Deepchecks_Suite",
        checks: Optional[List[BaseCheck]] = DEFAULT_CHECKS,
    ):
        super().__init__()
        self.suite_name = suite_name
        # TODO allow list of strings?
        self.checks = checks

    def _validate_arguments(self):
        """
        Check that basic requirements for the run of an evaluator are met.
        """
        check_requirements_deepchecks(self)

    def _setup(self):
        # Set artifacts

        # All artifacts are optional and thus any could be NoneType
        # Internal (lens) validation ensures that at least one artifact is valid
        self.model = self.model
        self.test_dataset = self.assessment_data
        self.train_dataset = self.training_data

    def evaluate(self):
        """
        Execute any data/model processing required for the evaluator.

        Populates the self.results object.

        Returns
        -------
        self
        """
        self._setup_deepchecks()
        self.run_suite()

        self.results = [DeepchecksContainer(self.suite_name, self.suite_results)]

        return self

    def _setup_deepchecks(self):
        if self.test_dataset:
            self.test_dataset = Dataset(
                df=self.test_dataset.X, label=self.test_dataset.y
            )

        if self.train_dataset:
            self.train_dataset = Dataset(
                df=self.train_dataset.X, label=self.train_dataset.y
            )

        if self.model:
            self.deepchecks_model = self.model.model_like

        self.suite = Suite(name=self.suite_name)
        for check in self.checks:
            self.suite.add(check)
            # doing this as a for-loop list seems to be the only way
            # deepchecks won't let you pass a whole list of checks, which is...silly?

    def run_suite(self):
        if self.train_dataset and self.test_dataset:
            self.suite_results = self.suite.run(
                train_dataset=self.train_dataset,
                test_dataset=self.test_dataset,
                model=self.model.model_like,
            )

        elif self.train_dataset:
            self.suite_results = self.suite.run(
                train_dataset=self.train_dataset, model=self.model.model_like
            )
        else:
            # Deepchecks expects the user to specify a train dataset if only a single
            # dataset is specified, even if that single dataset is supposed to be a test set
            # This doesn't really make sense and makes client code (like ours) less readable.
            # Nevertheless, there's no way around it.
            self.suite_results = self.suite.run(
                train_dataset=self.test_dataset, model=self.model.model_like
            )
