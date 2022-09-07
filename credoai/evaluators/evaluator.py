from abc import ABC, abstractmethod

from credoai.utils.common import NotRunError


class Evaluator(ABC):
    """
    Base abastract class for all lens evaluators.

    Defines basic functions required from any evaluator object.

    """

    @property
    def results(self):
        return None

    @property
    def artifacts(self):
        return []

    @property
    @abstractmethod
    def name(self):
        """Used to define a unique identifier for the specific evaluator"""
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
        # add arguments to properties of class
        self.__dict__.update(kwargs)
        self._validate_arguments()
        self._setup(**kwargs)
        return self

    @abstractmethod
    def _setup(self):
        pass

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

    @abstractmethod
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
    def _validate_arguments(self):
        """
        Check that basic requirements for the run of an evaluator are met.
        """
        pass
