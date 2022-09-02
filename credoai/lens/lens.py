from inspect import isclass
import inspect
import re
from typing import Union
import uuid

from absl import logging
from credoai.artifacts import Data
from credoai.artifacts import Model
from credoai.evaluators.evaluator import Evaluator
from credoai.utils.common import ValidationError
from credoai.lens.utils import log_command, build_list_of_evaluators


## TODO: Format the list of commands nicely

## TODO: Decide Metadata policy, connected to governance and evidence creation!


def set_logging_level(logging_level):
    """Alias for absl.logging.set_verbosity"""
    logging.set_verbosity(logging_level)


class Lens:
    def __init__(
        self,
        logging_level: Union[str, int] = "info",
        *,
        model: Model = (None,),
        data: Data = (None,),
        training_data: Data = (None,),
        pipeline: list = None,
    ) -> None:
        self.model = model
        self.assessment = data
        self.training = training_data
        self.assessment_plan = {}
        self.run_time = False
        self.gov = None
        self.pipeline = {}
        # If a list of steps is passed create the pipeline
        if pipeline:
            self._generate_pipeline(pipeline)

        self.pipeline_results = []
        self._validate()
        ## TODO: evaluate what library to use for logging
        set_logging_level(logging_level)
        self.command_list = []

    @log_command
    def add(self, evaluator, id: str = None, metadata: dict = None):
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
            id = f"{evaluator.name}_{str(uuid.uuid4())}"

        try:  # TODO: Create proper logging to capture validation issues
            evaluator_required_parameters = re.sub(
                "[\(\) ]", "", str(inspect.signature(evaluator))
            ).split(",")

            evaluator_arguments = {
                k: v
                for k, v in vars(self).items()
                if k in evaluator_required_parameters
            }

            self.pipeline[id] = {
                "evaluator": evaluator(**evaluator_arguments),
                "meta": metadata,
            }
        except Exception as e:
            print(e)
        return self

    @log_command
    def remove(self, id: str):
        # Find position
        del self.pipeline[id]
        return self

    @log_command
    def run(self):
        """
        Run the main loop across all the pipeline steps
        """
        if len(self.pipeline) == 0:
            print("Empty pipeline: proceeding with defaults...")
            self._generate_pipeline(build_list_of_evaluators())
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
        return self.command_list

    def push(self):
        """
        1. Convert internal evidence to platform evidence
        2. Push to platform, main code to do that should be in the governance folder
        """
        pass

    def _generate_pipeline(self, pipeline=None):
        """Automatically generate pipeline"""
        # Create pipeline from list of steps
        if pipeline:
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
                    raise ValidationError(
                        f"Id in step {step} must be a string, received {id}"
                    )
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
