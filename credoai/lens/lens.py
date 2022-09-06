from inspect import isclass
import inspect
import re
from typing import Union
import uuid

import logging
from credoai.artifacts import Data
from credoai.artifacts import Model
from credoai.evaluators.evaluator import Evaluator
from credoai.utils.common import ValidationError
from credoai.lens.utils import log_command, build_list_of_evaluators

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


## TODO: Decide Metadata policy, connected to governance and evidence creation!


class Lens:
    def __init__(
        self,
        *,
        model: Model = None,
        data: Data = None,
        training_data: Data = None,
        pipeline: list = None,
    ) -> None:
        self.model = model
        self.assessment = data
        self.training = training_data
        self.assessment_plan = {}
        self.run_time = False
        self.gov = None
        self.pipeline = {}
        self.command_list = []
        # If a list of steps is passed create the pipeline
        if pipeline:
            self._generate_pipeline(pipeline)

        self.pipeline_results = []
        self._validate()

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
            logging.info(f"Evaluator {evaluator.name} addedd to pipeline.")

        except ValidationError as e:
            logging.info(f"Evaluator {evaluator.name} NOT added to the pipeline: {e}")

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
