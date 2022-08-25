"""
Defines abstract base class for all CredoAssessments
"""

from abc import ABC, abstractmethod

import pandas as pd
from credoai.utils.common import ValidationError, wrap_list

# Class Docstring below is a template used for all assessments __init__
# Following this template helps for consistency, and filters down
# to reporting


class CredoAssessment(ABC):
    """{Short description of assessment}

    {Longer description of what the assessment does}

    Modules
    -------
    * credoai.modules.module1
    * credoai.modules.module2
    * etc...

    Requirements
    ------------
    {Describe requirements for CredoModel and CredoData
    as specified by AssessmentRequirements in plain english}}

    Parameters
    ----------
    ...
    """

    def __init__(self, name, module, requirements=None):
        """Abstract base class for all CredoAssessments

        Parameters
        ----------
        name : str
            Label of the assessment
        module : CredoModule
            CredoModule the Assessment builds
        requirements : AssessmentRequirements, optional
            Instantiation of functionality CredoModel and/or CredoData
            must define to run this assessment. If defined, enables
            automated validation and selection of assessment
        """
        self.name = name
        self.module = module
        self.initialized_module = None
        self.reporters = None
        if requirements is None:
            requirements = AssessmentRequirements()
        self.requirements = requirements
        # placeholders for artifact names
        self.model_name = None
        self.data_name = None
        self.training_data_name = None

    def __str__(self):
        data_name = self.data_name or "NA"
        return f"{self.name} - Dataset: {data_name}"

    @abstractmethod
    def init_module(self, *, model=None, data=None, training_data=None):
        """Initializes the assessment module

        Transforms CredoModel and CredoData into the proper form
        to create a runnable assessment.

        See the lens_customization notebook for examples

        Parameters
        ------------
        model : CredoModel, optional
        data : CredoData, optional
        training_data : CredoData, optional

        Example
        -----------
        def init_module(self, ...):
            y_pred = CredoModel.predict(CredoData.X)
            y = CredoData.y
            self.initialized_module = self.module(y_pred, y)

        """
        if model:
            self.model_name = model.name
        if data:
            self.data_name = data.name
        if training_data:
            self.training_data = training_data.name

    def init_reporters(self):
        """Initialize a reporter object"""
        pass

    def run(self):
        return self.initialized_module.run()

    def prepare_results(self, metadata=None, **kwargs):
        results = self.initialized_module.prepare_results(**kwargs)
        if results is None:
            return None
        results = self._standardize_prepared_results(results).fillna("NA")
        self._validate_results(results)
        # add metadata
        metadata = metadata or {}
        results = results.assign(**metadata)
        return results

    def get_description(self):
        return {"short": self.short_description, "long": self.long_description}

    def get_name(self):
        """Returns unique id for assessment

        For any model, an assessment is defined
        by the dataset
        """
        return self.name

    def get_results(self):
        return self.initialized_module.get_results()

    def get_reporters(self):
        """Gets reporters to visualize the assessment

        Does nothing if not overwritten
        """
        return wrap_list(self.reporters)

    def get_requirements(self):
        return self.requirements.get_requirements()

    def check_requirements(
        self, credo_model=None, credo_data=None, credo_training_data=None
    ):
        """
        Defines the functionality needed by the assessment

        Returns a list of functions that a CredoModel must
        instantiate to run. Defining this function supports
        automated assessment inference by Lens.

        Returns
        ----------
        credo.assessment.AssessmentRequirements
        """
        return self.requirements.check_requirements(
            credo_model, credo_data, credo_training_data
        )

    def _standardize_prepared_results(self, results):
        if type(results) == dict:
            results = pd.Series(results, name="value").to_frame()
        elif type(results) == pd.Series:
            results.name = "value"
            results = results.to_frame()
        elif type(results) == pd.DataFrame:
            pass
        else:
            raise TypeError("Results format not recognized")
        results.index.name = "metric_type"
        return results

    def _validate_results(self, results):
        if (
            type(results) != pd.DataFrame
            or results.index.name != "metric_type"
            or "value" not in results.columns
        ):
            raise ValidationError(
                f"{self.name} assessment results not in correct format"
            )


class AssessmentRequirements:
    def __init__(
        self,
        model_requirements=None,
        data_requirements=None,
        training_data_requirements=None,
        model_frameworks=None,
        model_types=None,
        target_types=None,
    ):
        """
        Defines requirements for an assessment

        Parameters
        ------------
        model_requirements : List(Union[List, str])
            Requirements as a list. Each element
            can be a single string representing a CredoModel
            attribute/function or a list of such attributes/functions.
            If a list, only one of those attributes/functions are
            needed to satisfy the requirements.
        {training_}data_requirements : List(Union[List, str])
            Requirements as a list. Each element
            can be a single string representing a CredoData
            attribute/function or a list of such attributes/functions.
            If a list, only one of those attributes/functions are
            needed to satisfy the requirements.
        model_frameworks : List(str)
            List of Model framework(s) required by assessment.
            Each element must be taken from list defined by
            credoai.utils.constants.SUPPORTED_FRAMEWORKS
        model_types : List(str)
            List of Model type(s) required by assessment.
            Each element must be taken from list defined by
            credoai.utils.constants.MODEL_TYPES
        target_types : List(str)
            List of Target type(s) required by assessment. Must be an output
            of sklearn.utils.multiclass.type_of_target
        """
        self.model_requirements = model_requirements or []
        self.data_requirements = data_requirements or []
        self.training_data_requirements = training_data_requirements or []
        self.model_frameworks = model_frameworks or []
        self.model_types = model_types or []
        self.target_types = target_types or []

    def check_requirements(
        self, credo_model=None, credo_data=None, credo_training_data=None
    ):
        # disqualify if the assessment does not require any of the artifacts provided
        if (
            (credo_model and not self.model_requirements)
            or (credo_data and not self.data_requirements)
            or (credo_training_data and not self.training_data_requirements)
        ):
            return False

        # check to make sure the artifact has the required functionality defined
        for artifact, requirements in [
            (credo_model, self.model_requirements),
            (credo_data, self.data_requirements),
            (credo_training_data, self.training_data_requirements),
        ]:
            if artifact:
                existing_keys = [
                    k for k, v in artifact.__dict__.items() if v is not None
                ]
                functionality = set(existing_keys)
            else:
                functionality = set()

            for requirement in requirements:
                if type(requirement) == str:
                    if not requirement in functionality:
                        return False
                else:
                    if not functionality.intersection(requirement):
                        return False
        # check frameworks
        if self.model_frameworks:
            if credo_model and credo_model.framework in self.model_frameworks:
                pass
            else:
                return False
        # check model type
        if self.model_types:
            if credo_model and credo_model.model_type in self.model_types:
                pass
            else:
                return False
        # check target type
        if self.target_types:
            for dataset in (credo_data, credo_training_data):
                if dataset and dataset.y_type in self.target_types:
                    pass
                else:
                    return False
        return True

    def get_requirements(self):
        return {
            "model_requirements": self.model_requirements,
            "data_requirements": self.data_requirements,
            "training_data_requirements": self.training_data_requirements,
        }
