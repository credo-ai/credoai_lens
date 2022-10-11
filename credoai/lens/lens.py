import uuid
from copy import deepcopy
from inspect import isclass
from typing import Dict, List, Optional, Tuple, Union

from credoai.artifacts import Data, Model
from credoai.evaluators.evaluator import Evaluator
from credoai.governance import Governance
from credoai.lens.pipeline_creator import PipelineCreator
from credoai.utils import ValidationError, flatten_list, global_logger

# Custom type
Pipeline = List[Union[Evaluator, Tuple[Evaluator, str, dict]]]


## TODO: Decide Metadata policy, connected to governance and evidence creation!


class Lens:
    def __init__(
        self,
        *,
        model: Model = None,
        assessment_data: Data = None,
        training_data: Data = None,
        pipeline: Pipeline = None,
        governance: Governance = None,
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
            second element is the step id (optional), third element (optional) is metadata (dict)
            associated to the step.
            - Evaluator. If the user does not intend to specify id or metadata, instantiated
            evaluators can be put directly in the list.
        governance : Governance, optional
            An instance of Credo AI's governance class. Used to handle interaction between
            Lens and the Credo AI Platform. Specifically, evidence requirements taken from
            policy packs defined on the platform will configure Lens, and evidence created by
            Lens can be exported to the platform.
        """
        self.model = model
        self.assessment_data = assessment_data
        self.training_data = training_data
        self.assessment_plan: dict = {}
        self.gov = governance
        self.pipeline: dict = {}
        self.command_list: list = []
        self.logger = global_logger
        self.pipeline_results: list = []
        if self.assessment_data and self.assessment_data.sensitive_features is not None:
            self.sens_feat_names = list(self.assessment_data.sensitive_features)
        else:
            self.sens_feat_names = []
        self._generate_pipeline(pipeline)
        # Can  pass pipeline directly
        self._validate()

    def __getitem__(self, stepname):
        return self.pipeline[stepname]

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
        eval_reqrd_params = evaluator.required_artifacts
        check_sens_feat = "sensitive_feature" in eval_reqrd_params
        check_data = "data" in eval_reqrd_params

        ## Validate basic requirements
        if check_sens_feat and not self.sens_feat_names:
            raise ValidationError(
                f"Evaluator {evaluator.name} requires sensitive features"
            )

        ## Define necessary arguments for evaluator
        evaluator_arguments = {
            k: v for k, v in vars(self).items() if k in eval_reqrd_params
        }

        ## Basic case: eval depends on specific datasets and not on sens feat
        try:
            if not check_data and not check_sens_feat:
                self._add(evaluator, id, metadata, evaluator_arguments)
                return self

            if check_sens_feat:
                features_to_eval = self.sens_feat_names
            else:
                features_to_eval = [self.sens_feat_names[0]]  # Cycle only once

            self._cycle_add_through_ds_feat(
                evaluator,
                id,
                check_sens_feat,
                check_data,
                evaluator_arguments,
                features_to_eval,
            )
        except ValidationError as e:
            self.logger.info(
                f"Evaluator {evaluator.name} NOT added to the pipeline: {e}"
            )
        return self

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

    def run(self):
        """
        Run the main loop across all the pipeline steps.
        """
        if self.pipeline == {}:
            raise RuntimeError("No evaluators were added to the pipeline.")
        for step, details in self.pipeline.items():
            self.logger.info(f"Running evaluation for step: {step}")
            details["evaluator"].evaluate()
            # Populate pipeline results
            self.pipeline_results.append(
                {"id": step, "results": details["evaluator"].results}
            )
        return self

    def send_to_governance(self, overwrite_governance=False):
        """
        Parameters
        ---------
        overwrite_governance : bool
            When adding evidence to a Governance object, whether to overwrite existing
            evidence or not, default False.
        """
        evidence = self.get_evidence()
        if self.gov:
            if overwrite_governance:
                self.gov.set_evidence(evidence)
                self.logger.info(
                    "Sending evidence to governance. Overwriting existing evidence."
                )
            else:
                self.gov.add_evidence(evidence)
                self.logger.info(
                    "Sending evidence to governance. Adding to existing evidence."
                )
        else:
            raise ValidationError(
                "No governance object exists to update."
                " Call lens.set_governance to add a governance object."
            )
        return self

    def get_evidence(self):
        """
        Converts evaluator results to evidence for the platform from the pipeline results.

        Return
        ------
        List of Evidence
        """
        all_results = flatten_list([x["results"] for x in self.pipeline_results])
        evidences = []
        for result in all_results:
            evidences += result.to_evidence()
        return evidences

    def get_results(self) -> Dict:
        """
        Extract results from the pipeline output.

        Returns
        -------
        Dict
            The format of the dictionary is Pipeline step id: results
        """
        res = {x["id"]: [i.df for i in x["results"]] for x in self.pipeline_results}
        return res

    def get_command_list(self):
        return print("\n".join(self.command_list))

    def get_evaluators(self):
        return [i["evaluator"] for i in self.pipeline.values()]

    def print_results(self):
        results = self.get_results()
        for key, val in results.items():
            print(f"Evaluator: {key}\n")
            for i in val:
                print(i)
                print()
            print()

    def set_governance(self, governance: Governance):
        self.gov = governance

    def _add(
        self,
        evaluator: Evaluator,
        id: Optional[str],
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
        if id is None:
            id = f"{evaluator.name}_{str(uuid.uuid4())[0:6]}"

        ## Attempt pipe addition

        self.pipeline[id] = {
            "evaluator": evaluator(**evaluator_arguments),
            "meta": metadata,
        }

        # Create logging message
        logger_message = f"Evaluator {evaluator.name} added to pipeline. "
        if metadata is not None:
            if "dataset" in metadata:
                logger_message += f"Dataset used: {metadata['dataset']}. "
            if "sensitive_feature" in metadata:
                logger_message += f"Sensitive feature: {metadata['sensitive_feature']}"
        self.logger.info(logger_message)

    def _cycle_add_through_ds_feat(
        self,
        evaluator,
        id,
        check_sens_feat,
        check_data,
        evaluator_arguments,
        features_to_eval,
    ):
        for feat in features_to_eval:
            if check_data:
                available_datasets = [
                    n for n, a in vars(self).items() if "data" in n if a
                ]
                for dataset in available_datasets:
                    labels = {"sensitive_feature": feat} if check_sens_feat else {}
                    labels["dataset"] = dataset
                    evaluator_arguments["data"] = vars(self)[dataset]
                    self.change_sens_feat_view(evaluator_arguments, feat)
                    self._add(deepcopy(evaluator), id, labels, evaluator_arguments)
            else:
                self.change_sens_feat_view(evaluator_arguments, feat)
                self._add(
                    deepcopy(evaluator),
                    id,
                    {"sensitive_feature": feat},
                    evaluator_arguments,
                )
        return self

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
        if pipeline is None:
            if self.gov:
                self.logger.info("Empty pipeline: generating from governance.")
                pipeline = PipelineCreator.generate_from_governance(self.gov)
            else:
                return
        # Create pipeline from list of steps
        for step in pipeline:
            if not isinstance(step, tuple):
                step = (step,)
            evaltr, id, meta = self._consume_pipeline_step(step)
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
        if not isinstance(self.assessment_data, Data):
            raise ValidationError(
                "Assessment data should inherit from credoai.artifacts.Data"
            )
        if not (isinstance(self.training_data, Data) or self.training_data is None):
            raise ValidationError(
                "Assessment data should inherit from credoai.artifacts.Data"
            )

        if self.assessment_data is not None and self.training_data is not None:
            if (
                self.assessment_data.sensitive_features is not None
                and self.training_data.sensitive_features is not None
            ):
                if len(self.assessment_data.sensitive_features.shape) != len(
                    self.training_data.sensitive_features.shape
                ):
                    raise ValidationError(
                        "Sensitive features should have the same shape across assessment and training data"
                    )

    @staticmethod
    def _consume_pipeline_step(step):
        def safe_get(step, index):
            return (step[index : index + 1] or [None])[0]

        evaltr = safe_get(step, 0)
        id = safe_get(step, 1)
        meta = safe_get(step, 2)
        return evaltr, id, meta

    @staticmethod
    def change_sens_feat_view(evaluator_arguments: Dict[str, Data], feat: str):
        for artifact in evaluator_arguments.values():
            if getattr(artifact, "active_sens_feat", False):
                artifact.active_sens_feat = feat
        return evaluator_arguments
