from typing import List, Type, Union
import uuid

from absl import logging
from credoai.artifacts import Data
from credoai.artifacts import Model
from credoai.evaluators.evaluator import Evaluator
from credoai.utils.common import ValidationError

## TODO: Define what format is the assessment plan coming from
## the platform.

## TODO: Display policy checklists -> decide if necessary


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
    ) -> None:
        self.model = model
        self.assessment_dataset = data
        self.training_dataset = training_data
        self.assessment_plan = {}
        self.run_time = False
        self.gov = None
        self.pipeline = {}
        self.pipeline_results = []
        self._validate()
        ## TODO: evaluate what library to use for logging
        set_logging_level(logging_level)

    def __call__(self):
        self._create_pipeline()
        self.run()

    def add(self, evaluator, id: str = None, metadata: dict = None):
        ## Validate same identifier doesn't exist already
        if id in self.pipeline:
            raise ValueError(
                f"An evaluator with id: {id} is already in the pipeline. Id has to be unique"
            )

        if not isinstance(evaluator, Evaluator):
            ## TODO: Make sure instance is correct after evaluator type is defined
            raise TypeError(
                f"Evaluator has to be of type evaluator... not {type(evaluator)}"
            )
        if id is None:
            id = f"{evaluator.name}_{str(uuid.uuid4())}"

        self.pipeline[id] = {
            "evaluator": evaluator(
                model=self.model,
                assessment=self.assessment_dataset,
                training=self.training_dataset,
            ),
            "meta": metadata,
        }
        return self

    def remove(self, id: str):
        # Find position
        del self.pipeline[id]
        return self

    def run(self, pipeline=None):
        """
        Run the main loop across all the pipeline steps
        """
        # Can  pass pipeline directly
        if pipeline:
            self.pipeline = pipeline
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

    def push(self):
        """
        1. Convert internal evidence to platform evidence
        2. Push to platform, main code to do that should be in the governance folder
        """
        pass

    def _create_pipeline(self):
        """Automatically generate pipeline"""
        pass

    def _validate(self):
        """
        Validate arguments passed to Lens. All checks should be here

        Raises
        ------
        ValidationError
        """
        if self.assessment_dataset and self.training_dataset:
            if self.assessment_dataset == self.training_dataset:
                raise ValidationError(
                    "Assessment dataset and training dataset should not be the same!"
                )
