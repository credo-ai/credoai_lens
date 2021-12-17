"""
Defines abstract base class for all CredoAssessments
"""

from abc import ABC, abstractmethod
from credoai.utils.common import ValidationError
import pandas as pd


class CredoAssessment(ABC):
    """Abstract base class for all CredoAssessments

    Parameters
    ----------
    name : str
        Label of the assessment
    module : CredoModule
        CredoModule the Assessment builds
    requirements : AssessmentRequirements, optional
        Instantiation of funtionality CredoModel and/or CredoData
        must define to run this asssesment. If defined, enables
        automated validation and selection of asssessment
    """

    def __init__(self, name, module, requirements=None):
        self.name = name
        self.module = module
        self.initialized_module = None
        if requirements is None:
            requirements = AssessmentRequirements()
        self.requirements = requirements

    @abstractmethod
    def init_module(self, *, manifest=None, model=None, data=None):
        """ Initializes the assessment module

        Transforms the manifest, CredoModel and CredoData into the proper form
        to create a runnable assessment.

        See the lens_customization notebook for examples

        Parameters
        ------------
        manifest : dict
            dictionary containing kwargs for the module defined by the manifest
        model : CredoModel, optional
        data : CredoData, optional

        Example:
        def build(self, ...):
            y_pred = CredoModel.pred_fun(CredoData.X)
            y = CredoData.y
            self.initialized_module = self.module(y_pred, y)

        """
        pass

    def run(self, **kwargs):
        return self.initialized_module.run(**kwargs)

    def prepare_results(self, metadata=None, **kwargs):
        results = self.initialized_module.prepare_results(**kwargs)
        results = self._standardize_results(results)
        self._validate_results(results)
        # add metadata
        metadata = metadata or {}
        metadata['assessment'] = self.name
        results = results.assign(**metadata)
        return results

    def _standardize_results(self, results):
        if type(results) == dict:
            results = pd.Series(results, name='value').to_frame()
            results.index.name = 'metric'
        elif type(results) == pd.Series:
            results.index.name = 'metric'
            results.name = 'value'
            results = results.to_frame()
        elif type(results) == pd.DataFrame():
            pass
        else:
            raise TypeError("Results format not recognized")
        return results

    def _validate_results(self, results):
        if (type(results) != pd.DataFrame
            or results.index.name != 'metric'
                or 'value' not in results.columns):
            raise ValidationError(
                f'{self.name} assessment results not in correct format')

    def check_requirements(self,
                           credo_model=None,
                           credo_data=None):
        """
        Defines the functionality needed by the assessment

        Returns a list of functions that a CredoModel must
        instantiate to run. Defining this function supports
        automated assessment inference by Lens. 

        Returns
        ----------
        credo.asseesment.AssessmentRequirements
        """
        return self.requirements.check_requirements(credo_model,
                                                    credo_data)

    def get_requirements(self):
        return self.requirements.get_requirements()


class AssessmentRequirements:
    def __init__(self,
                 model_requirements=None,
                 data_requirements=None):
        """
        Defines requirements for an assessment

        Parameters
        ------------
        requirements : List(Union[List, str])
            Requirements as a list. Each element
            can be a single string representing a CredoModel
            function or a list of such functions. If a list,
            only one of those functions are needed to satisfy
            the requirements.
        """
        self.model_requirements = model_requirements or []
        self.data_requirements = data_requirements or []

    def check_requirements(self, credo_model=None, credo_data=None):
        for artifact, requirements in \
            [(credo_model, self.model_requirements),
             (credo_data, self.data_requirements)]:
            if artifact:
                existing_keys = [k for k, v in artifact.__dict__.items()
                                 if v is not None]
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
        return True

    def get_requirements(self):
        return {'model_requirements': self.model_requirements,
                'data_requirements': self.data_requirements}
