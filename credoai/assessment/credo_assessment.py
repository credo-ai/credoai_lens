"""
Defines abstract base class for all CredoAssessments
"""

from abc import ABC, abstractmethod
from credoai.utils.common import ValidationError
import pandas as pd

# Class Docstring below is a template used for all assessments __init__
# Following this template helps for consistency, and filters down
# to reporting


class CredoAssessment(ABC):
    """{Short description of assessment}

    {Longer escription of what the assessment does}

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

    def __init__(self, name, module, requirements=None,
                 short_description=None, long_description=None):
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
        short_description : str
            Short description of assessment functionality. If None
            will default to first line of class docstring
        long_description : str
            Long description of assessment functionality. If None
            will default to first line of class docstring
        """
        self.name = name
        self.module = module
        self.initialized_module = None
        self.report = None
        if requirements is None:
            requirements = AssessmentRequirements()
        self.requirements = requirements
        # descriptions, automatically parsed fro docstring if not set
        self.short_description = short_description
        self.long_description = long_description
        self._set_description_from_doc()

    @abstractmethod
    def init_module(self, *, model=None, data=None):
        """ Initializes the assessment module

        Transforms CredoModel and CredoData into the proper form
        to create a runnable assessment.

        See the lens_customization notebook for examples

        Parameters
        ------------
        model : CredoModel, optional
        data : CredoData, optional

        Example
        -----------
        def init_module(self, ...):
            y_pred = CredoModel.pred_fun(CredoData.X)
            y = CredoData.y
            self.initialized_module = self.module(y_pred, y)

        """
        pass

    def run(self, **kwargs):
        return self.initialized_module.run(**kwargs)

    def prepare_results(self, metadata=None, **kwargs):
        results = self.initialized_module.prepare_results(**kwargs)
        results = self._standardize_prepared_results(results)
        self._validate_results(results)
        # add metadata
        metadata = metadata or {}
        metadata['assessment'] = self.name
        results = results.assign(**metadata)
        # return results (and ensure no NaN floats remain)
        return results.fillna('NA')

    def get_description(self):
        return {'short': self.short_description,
                'long': self.long_description}

    def get_results(self):
        return self.initialized_module.get_results()

    def get_reporter(self):
        """Gets reporter to visualize the assessment

        Does nothing if not overwritten
        """
        pass   

    def _set_description_from_doc(self):
        docs = self.__doc__
        # underline title of next section
        try:
            description = docs[:(docs.index('---')-5)]
        except ValueError:
            description = docs
        # remove last line (title of next section)
        description = description[:description.rfind('\n')].lstrip()
        short = description.split('\n')[0]
        long = description[len(short)+2:]
        if self.short_description is None:
            self.short_description = short
        if self.long_description is None:
            self.long_description = long

    def _standardize_prepared_results(self, results):
        if type(results) == dict:
            results = pd.Series(results, name='value').to_frame()
            results.index.name = 'metric_type'
        elif type(results) == pd.Series:
            results.index.name = 'metric_type'
            results.name = 'value'
            results = results.to_frame()
        elif type(results) == pd.DataFrame:
            pass
        else:
            raise TypeError("Results format not recognized")
        return results

    def _validate_results(self, results):
        if (type(results) != pd.DataFrame
            or results.index.name != 'metric_type'
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
