from inspect import isclass
import inspect
import re
from typing import List, Union
import uuid

import logging
from credoai.artifacts import Data
from credoai.artifacts import Model
from credoai.evaluators.evaluator import Evaluator
from credoai.utils.common import ValidationError
from credoai.lens.utils import log_command, build_list_of_evaluators

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
# Custom type
Pipeline = List[Union[Evaluator, tuple[Evaluator, str, dict]]]


## TODO: Decide Metadata policy, connected to governance and evidence creation!


class Lens:
    def __init__(
        self,
        *,
        model: Model = None,
        data: Data = None,
        training_data: Data = None,
        pipeline: Pipeline = None,
    ) -> None:
        """
        Initializer for the Lens class.

        Parameters
        ----------
        model : Model, optional
            Credo Model, by default None
        data : Data, optional
            Assessment/test data, by default None
        training_data : Data, optional
            Training data, extra dataset used by some of the evaluators, by default None
        pipeline : Pipeline_type, optional, default None
            User can add a pipeline using a list of steps. Steps can be in 2 formats:
            - tuple: max length = 3. First element is the instantiated evaluator,
            second element is the step id (optional), third elemnt(optional) is metadata (dict)
            associated to the step.
            - Evaluator. If the user does not intend to specify id or metadata, instantiated
            evaluators can be put directly in the list.
        """
        self.model = model
        self.assessment = data
        self.training = training_data
        self.assessment_plan: dict = {}
        self.gov = None
        self.pipeline: dict = {}
        self.command_list: list = []
        self.logger = logging.getLogger(self.__class__.__name__)
        # If a list of steps is passed create the pipeline
        if pipeline:
            self._generate_pipeline(pipeline)
        self.pipeline_results: list = []
        self._validate()

    @log_command
    def add(self, evaluator: Evaluator, id: str = None, metadata: dict = None):
        """
        Add a single step to the pipeline.

        The function also passess extra arguments to the instantiated evaluator via
        a call to the __call__ method of the evaluator. Only the arguments required
        by the evaluator are provided.

        Parameters
        ----------
        evaluator : Evaluator
            Instantiated Credo Evaluator.
        id : str, optional
            A string to identify the step. If not provided one is randomly
            generated, by default None
        metadata : dict, optional
            Any metadata associated to the step the user wants to add, by default None

        Raises
        ------
        ValueError
            Ids cannot be duplicated in the pipeline.
        TypeError
            The first object passed to the add method needs to be a Credo Evaluator.
        """
        ## Validate same identifier doesn't exist already
        if id in self.pipeline:
            raise ValueError(
                f"An evaluator with id: {id} is already in the pipeline. Id has to be unique"
            )

        if not isinstance(evaluator, Evaluator):
            raise TypeError(
                f"Evaluator has to be of type evaluator, received {type(evaluator)}"
            )
        if id is None:
            ## TODO: Check if it makes sense to hash arguments to ensure uniqueness
            id = f"{evaluator.name}_{str(uuid.uuid4())}"

        ## Define necessary arguments for evaluator
        evaluator_required_parameters = re.sub(
            "[\(\) ]", "", str(inspect.signature(evaluator))
        ).split(",")

        evaluator_arguments = {
            k: v for k, v in vars(self).items() if k in evaluator_required_parameters
        }
        ## Attampt pipe addition
        try:
            self.pipeline[id] = {
                "evaluator": evaluator(**evaluator_arguments),
                "meta": metadata,
            }
            self.logger.info(f"Evaluator {evaluator.name} addedd to pipeline.")
        except ValidationError as e:
            self.logger.info(
                f"Evaluator {evaluator.name} NOT added to the pipeline: {e}"
            )

    @log_command
    def remove(self, id: str):
        """
        Remove a step from the pipeline based on the id.

        Parameters
        ----------
        id : str
            Id of the step to remove
        """
        # Find position
        del self.pipeline[id]
        return self

    @log_command
    def run(self):
        """
        Run the main loop across all the pipeline steps.
        """
        if len(self.pipeline) == 0:
            logging.info("Empty pipeline: proceeding with defaults...")
            all_evaluators = build_list_of_evaluators()
            self._generate_pipeline(all_evaluators)
        # Can  pass pipeline directly
        for step, details in self.pipeline.items():
            details["evaluator"].evaluate()
            # Populate pipeline results
            self.pipeline_results.append(
                {"id": step, "results": details["evaluator"].results}
            )
        return self

    def get_results(self):
        """
        Collect all the results as they are coming directly from the individual evaluations.
        """
        return self.pipeline_results

    def get_command_list(self):
        return print("\n".join(self.command_list))

    def push(self):
        """
        Placeholder!
        1. Convert internal evidence to platform evidence (unless this is already part of
        each specific evaluator)
        2. Push to platform, main code to do that should be in the governance folder
        """
        pass

    def pull(self):
        """
        Placeholder!
        1. Gets the assessment plan from the platform.
        """
        pass

    def _generate_pipeline(self, pipeline):
        """
        Populates the pipeline starting from a list of steps.

        Parameters
        ----------
        pipeline : Pipeline_type, optional
            List of steps, by default None

        Raises
        ------
        ValidationError
            Each evaluator in a step needs to be already instantiated by the user.
        ValueError
            Id needs to be a string.
        """
        # Create pipeline from list of steps
        for step in pipeline:
            if not isinstance(step, tuple):
                step = (step,)
            evaltr = self._get_step_param(step, 0)
            id = self._get_step_param(step, 1)
            meta = self._get_step_param(step, 2)
            if isclass(evaltr):
                raise ValidationError(
                    f"Evaluator in step {step} needs to be instantiated"
                )
            if not (isinstance(id, str) or id is None):
                raise ValueError(f"Id in step {step} must be a string, received {id}")
            self.add(evaltr, id, meta)

        return self

    def _validate(self):
        """
        Validate arguments passed to Lens. All checks should be here

        Raises
        ------
        ValidationError
        """
        if self.assessment and self.training:
            if self.assessment == self.training:
                raise ValidationError(
                    "Assessment dataset and training dataset should not be the same"
                )

    @staticmethod
    def _get_step_param(step, pos):
        try:
            return step[pos]
        except IndexError:
            return None
