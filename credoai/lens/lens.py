import logging
import re
from typing import Dict, List, Optional, Union
import uuid
from inspect import isclass

from credoai.artifacts import Data, Model
from credoai.evaluators.evaluator import Evaluator
from credoai.lens.utils import build_list_of_evaluators, log_command
from credoai.utils.common import ValidationError

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
# Custom type
Pipeline = List[Union[Evaluator, tuple[Evaluator, str, dict]]]


## TODO: Decide Metadata policy, connected to governance and evidence creation!


class Lens:
    def __init__(
        self,
        *,
        model: Model = None,
        assessment_data: Data = None,
        training_data: Data = None,
        pipeline: Pipeline = None,
    ) -> None:
        """
        Initializer for the Lens class.

        Parameters
        ----------
        model : Model, optional
            Credo Model, by default None
        assessment_data : Data, optional
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
        self.assessment_data = assessment_data
        self.training_data = training_data
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

    def __getitem__(self, stepname):
        return self.pipeline[stepname]

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

        if id is None:
            ## TODO: Check if it makes sense to hash arguments to ensure uniqueness
            id = f"{evaluator.name}_{str(uuid.uuid4())[0:6]}"

        ## Define necessary arguments for evaluator
        evaluator_required_parameters = evaluator.required_artifacts

        evaluator_arguments = {
            k: v for k, v in vars(self).items() if k in evaluator_required_parameters
        }
        self._add(evaluator, id, metadata, evaluator_arguments)

        return self

    def _add(
        self,
        evaluator: Evaluator,
        id: str,
        metadata: Optional[dict],
        evaluator_arguments: dict,
    ):
        """
        Add a specific step while handling errors.

        Parameters
        ----------
        evaluator : Evaluator
            Instantiated evaluator
        id : str
            Step identifier
        evaluator_arguments : dict
            Arguments needed for the specific evaluator
        metadata : dict, optional
            Any Metadata to associate to the evaluator, by default None
        """

        ## Attempt pipe addition
        try:
            self.pipeline[id] = {
                "evaluator": evaluator(**evaluator_arguments),
                "meta": metadata,
            }
            self.logger.info(f"Evaluator {evaluator.name} added to pipeline.")
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

    def get_evidences(self):
        """
        Create evidences for the platform from the pipeline results.
        """
        labels = {
            "model_id": self.model.name if self.model else None,
            "dataset_name": self.assessment_data.name if self.assessment_data else None,
            "sensitive_features": [
                x for x in self.assessment_data.sensitive_features.columns
            ],
        }
        all_evidences = [x["results"] for x in self.pipeline_results]

        all_evidences = self.flatten_list(all_evidences)
        all_evidences = [x.to_evidence("id", **labels) for x in all_evidences]
        all_evidences = self.flatten_list(all_evidences)

        return [x.struct() for x in all_evidences]

    def get_results(self) -> Dict:
        """
        Extract results from the pipeline output.

        Returns
        -------
        Dict
            The format of the dictionaryu is Pipeline step id: results
        """

        res = {}
        for x in self.pipeline_results:
            if isinstance(x["results"], list):
                value = [i.df for i in x["results"]]
            else:
                value = x["results"].df
            res[x["id"]] = value
        return res

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
        if self.assessment_data and self.training_data:
            if self.assessment_data == self.training_data:
                raise ValidationError(
                    "Assessment dataset and training dataset should not be the same"
                )

    @staticmethod
    def _get_step_param(step, pos):
        try:
            return step[pos]
        except IndexError:
            return None

    @staticmethod
    def flatten_list(mixl):
        flattened = []
        for i in mixl:
            if hasattr(i, "__iter__"):
                for j in i:
                    flattened.append(j)
            else:
                flattened.append(i)
        return flattened
