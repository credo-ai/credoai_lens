import inspect
from abc import ABC, abstractmethod

from credoai.evidence import EvidenceContainer
from credoai.utils.common import NotRunError, ValidationError


class Evaluator(ABC):
    """
    Base abastract class for all lens evaluators.

    Defines basic functions required from any evaluator object.

    """

    def __init__(self):
        self._results = None
        self.artifact_keys = []

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, results):
        if not isinstance(results, list):
            raise ValidationError("Results must be a list")
        for result in results:
            if not isinstance(result, EvidenceContainer):
                raise ValidationError("All results must be EvidenceContainers")
        self._results = results

    @property
    @abstractmethod
    def name(self):
        """Used to define a unique identifier for the specific evaluator"""
        pass

    @property
    @abstractmethod
    def required_artifacts(self):
        pass

    def __call__(self, **kwargs):
        """
        This method is used to pass the model, assessment_dataset and training_dataset
        to instantiated evaluator.

        After objects are passed, it performs arguments validation and calls _setup

        >>> pipeline = Lens(model = model, assessment_data = dataset1)

        where a group of arguments shareable across multiple evaluators is passed.
        This method inside a specific evaluator takes the required arguments and
        makes them available to the evaluator instance.

        Requirements
        -------------
        _shared_arg_assignment requires explicitly named arguments.

        Returns
        -------
        self

        Implementation template
        -----------------------
        The following code template provides an example of what the internal of this
        method could look like:

        >>> self.model = kwargs['model']
        >>> self.assessment_dataset = kwargs['assessment_dataset']

        where model and assessment_dataset are Lens() arguments.

        """
        self._init_artifacts(kwargs)
        self._validate_arguments()
        self._setup()
        return self

    @abstractmethod
    def evaluate(self):
        """
        Execute any data/model processing required for the evaluator.

        Populates the self.results object.

        Returns
        -------
        self
        """
        return self

    def get_container_info(self, labels: dict = None, metadata: dict = None):
        info = self._base_container_info()
        if labels:
            info["labels"].update(labels)
        if metadata:
            info["metadata"].update(metadata)
        return info

    def _base_container_info(self):
        return {"labels": {"evaluator": self.name}, "metadata": self._get_artifacts()}

    def _get_artifacts(self):
        artifacts = {}
        for k in self.artifact_keys:
            try:
                artifacts[k] = self.__dict__[k].name
            except AttributeError:
                pass
        return artifacts

    def _init_artifacts(self, artifacts):
        """Adds artifacts to evaluator object

        Parameters
        ----------
        artifacts : dict
            Dictionary of artifacts, e.g. {'model': Model}
        """
        self.artifact_keys = list(artifacts.keys())
        self.__dict__.update(artifacts)

    def _prepare_results(self):
        """
        Transforms the results of the evaluation in internal evidences.

        Returns
        --------
        Internal evidence type
        """
        if self.results is not None:
            # prepare results code
            pass
        else:
            raise NotRunError(
                "Results not created yet. Call evaluate with the appropriate method"
            )

    @abstractmethod
    def _setup(self):
        pass

    @abstractmethod
    def _validate_arguments(self):
        """
        Check that basic requirements for the run of an evaluator are met.
        """
        pass
