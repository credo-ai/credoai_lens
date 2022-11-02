from abc import ABC, abstractmethod

from credoai.evidence import EvidenceContainer
from credoai.utils import global_logger
from credoai.utils.common import NotRunError, ValidationError


class Evaluator(ABC):
    """
    Base abstract class for all lens evaluators.

    Defines basic functions required from any evaluator object.

    """

    def __init__(self):
        self._results = None
        self.artifact_keys = []
        self.logger = global_logger
        self.metadata = {}

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def results(self):
        if self._results:
            return self._results
        else:
            raise NotRunError(
                "No results available, please call the method: 'evaluate'."
            )

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
    def required_artifacts(self):
        pass

    def __call__(self, **kwargs):
        """
        This method is used to pass the model, assessment_data and training_data
        artifacts to instantiated evaluator.

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
        meta = {**self.metadata, **self._get_artifacts()}
        labels = {"evaluator": self.name}
        if "dataset_type" in meta:
            labels["dataset_type"] = meta["dataset_type"]
        return {"labels": labels, "metadata": meta}

    def _get_artifacts(self):
        artifacts = {}
        save_keys = {
            "model": "model_name",
            "data": "dataset_name",
            "assessment_data": "assessment_dataset_name",
            "training_data": "training_dataset_name",
        }
        for k in self.artifact_keys:
            save_key = save_keys.get(k, k)
            try:
                artifacts[save_key] = self.__dict__[k].name
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

    @abstractmethod
    def _setup(self):
        pass

    @abstractmethod
    def _validate_arguments(self):
        """
        Check that basic requirements for the run of an evaluator are met.
        """
        pass
