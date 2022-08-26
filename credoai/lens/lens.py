from dataclasses import dataclass
import inspect
from typing import List, Type, Union
from credoai.artifacts import CredoGovernance, CredoModel, CredoData
from credoai.utils.common import (
    ValidationError,
)
from absl import logging
from credoai.evaluators.evaluator import Evaluator

## TODO: Define what format is the assessment plan coming from
## the platform.

## TODO: Display policy checklists -> decide if necessary

## TODO: dev mode? -> decide if necessary

## TODO: compare with original lens and scope difference


def set_logging_level(logging_level):
    """Alias for absl.logging.set_verbosity"""
    logging.set_verbosity(logging_level)


@dataclass
class Lens:
    def __init__(
        self,
        model: CredoModel = (None,),
        data: CredoData = (None,),
        training_data: CredoData = (None,),
        governance: Union[CredoGovernance, str] = (None,),
        logging_level: Union[str, int] = "info",
    ) -> None:
        self.model = model
        self.assessment_dataset = data
        self.training_dataset = training_data
        self.assessment_plan = {}
        self.run_time = False
        self.gov = None
        self.governance = governance
        self.pipeline = []
        self.pipeline_steps = []
        self.pipeline_results = []
        self._validate()
        ## TODO: evaluate what library to use for logging
        set_logging_level(logging_level)

    def add(self, evaluator, id: str, metadata: dict = None):
        ## Validate same identifier doesn't exist already
        if id in self.pipeline_steps:
            raise ValueError(
                f"An evaluator with id: {id} is already in the pipeline. Id has to be unique"
            )

        if not isinstance(evaluator, Evaluator):
            ## TODO: Make sure instance is correct after evaluator type is defined
            raise TypeError(
                f"Evaluator has to be of type evaluator...not {type(evaluator)}"
            )
        self.pipeline.append({"id": id, "evaluator": evaluator, "meta": metadata})
        self.pipeline_steps.append(id)
        return self

    def remove(self, id: str):
        # Find position
        step_to_remove = self.pipeline_steps.index(id)
        # Remove from both lists
        self.pipeline.pop(step_to_remove)
        self.pipeline_steps.pop(step_to_remove)
        return self

    def run(self):
        """
        Run the main loop across all the pipeline steps
        """
        for step in self.pipeline_steps:
            ## TODO: formalize when internal evidence class is completed how to pass metadata
            step["evaluator"].evaluate()
            ## TODO check results is the actual attribute in the evaluator class
            # Populate pipeline results
            self.pipeline_results.append(
                {"id": step["id"], "results": step["evaluator"].results}
            )
        return self

    def get_results(self):
        """
        Collect all the results as they are coming directly from the individual evaluations.
        """
        return self.pipeline_results

    def push(self):
        """
        1. Convert internal evidence to platform evidence
        2. Push to platform, main code to do that should be in the governance folder
        """
        pass

    def _validate(self):
        """
        Validate validity of arguments passed to Lens. All checks should be here

        Raises
        ------
        ValidationError
            _description_
        """
        if self.assessment_dataset and self.training_dataset:
            if self.assessment_dataset == self.training_dataset:
                raise ValidationError(
                    "Assessment dataset and training dataset should not be the same!"
                )
