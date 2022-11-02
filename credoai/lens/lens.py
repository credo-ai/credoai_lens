"""
Main orchestration module handling running evaluators on AI artifacts
"""

from copy import deepcopy
from dataclasses import dataclass
from inspect import isclass
from typing import Dict, List, Optional, Tuple, Union

from credoai.artifacts import Data, Model
from credoai.evaluators.evaluator import Evaluator
from credoai.governance import Governance
from credoai.lens.pipeline_creator import PipelineCreator
from credoai.utils import ValidationError, check_subset, flatten_list, global_logger

# Custom type
Pipeline = List[Union[Evaluator, Tuple[Evaluator, str, dict]]]


## TODO: Decide Metadata policy, connected to governance and evidence creation!


@dataclass
class PipelineStep:
    evaluator: Evaluator
    metadata: Optional[dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # TODO: keeping track of metadata somewhat unnecessarily. Could just add the metadata
        # directly to pipeline
        self.evaluator.metadata = self.metadata
        self.metadata["evaluator"] = self.evaluator.name

    def check_match(self, metadata):
        """Return true if metadata is a subset of pipeline step's metadata"""
        return check_subset(metadata, self.metadata)


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
            - tuple: max length = 2. First element is the instantiated evaluator,
             second element (optional) is metadata (dict) associated to the step.
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
        self.gov = None
        self.pipeline: list = []
        self.logger = global_logger
        if self.assessment_data and self.assessment_data.sensitive_features is not None:
            self.sens_feat_names = list(self.assessment_data.sensitive_features)
        else:
            self.sens_feat_names = []
        self._add_governance(governance)
        self._generate_pipeline(pipeline)
        # Can  pass pipeline directly
        self._validate()

    def add(self, evaluator: Evaluator, metadata: dict = None):
        """
        Add a single step to the pipeline.

        The function also passes extra arguments to the instantiated evaluator via
        a call to the __call__ method of the evaluator. Only the arguments required
        by the evaluator are provided.

        Parameters
        ----------
        evaluator : Evaluator
            Instantiated Credo Evaluator.
        metadata : dict, optional
            Any metadata associated to the step the user wants to add, by default None

        Raises
        ------
        ValueError
            Ids cannot be duplicated in the pipeline.
        TypeError
            The first object passed to the add method needs to be a Credo Evaluator.
        """
        step = PipelineStep(evaluator, metadata)

        eval_reqrd_params = step.evaluator.required_artifacts
        check_sens_feat = "sensitive_feature" in eval_reqrd_params
        check_data = "data" in eval_reqrd_params

        ## Validate basic requirements
        if check_sens_feat and not self.sens_feat_names:
            raise ValidationError(
                f"Evaluator {step.evaluator.name} requires sensitive features"
            )

        ## Define necessary arguments for evaluator
        evaluator_arguments = {
            k: v for k, v in vars(self).items() if k in eval_reqrd_params
        }

        ## Basic case: eval depends on specific datasets and not on sens feat
        try:
            if not check_data and not check_sens_feat:
                self._add(step, evaluator_arguments)
                return self

            if check_sens_feat:
                features_to_eval = self.sens_feat_names
            else:
                features_to_eval = [self.sens_feat_names[0]]  # Cycle only once

            self._cycle_add_through_ds_feat(
                step,
                check_sens_feat,
                check_data,
                evaluator_arguments,
                features_to_eval,
            )
        except ValidationError as e:
            self.logger.info(
                f"Evaluator {step.evaluator.name} NOT added to the pipeline: {e}"
            )
        return self

    def remove(self, index: int):
        """
        Remove a step from the pipeline based on the id.

        Parameters
        ----------
        index : int
            Index of the step to remove
        """
        # Find position
        del self.pipeline[index]
        return self

    def run(self):
        """
        Run the main loop across all the pipeline steps.
        """
        if self.pipeline == []:
            raise RuntimeError("No evaluators were added to the pipeline.")
        for step in self.pipeline:
            self.logger.info(f"Running evaluation for step: {step}")
            step.evaluator.evaluate()
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

    def get_datasets(self):
        return {
            name: data
            for name, data in vars(self).items()
            if "data" in name and data is not None
        }

    def get_evidence(self, evaluator_name=None, metadata=None):
        """
        Extract evidence from pipeline steps. Uses get_pipeline to determine to subset
        of pipeline steps to use

        Parameters
        ----------
        evaluator_name : str
            Name of evaluator to use to filter results. Must match the class name of an evaluator.
            Passed to `get_pipeline`
        metadata : dict
            Dictionary of evaluator metadata to filter results. Will return pipeline results
            whose metadata is a superset of the passed metadata. Passed to `get_pipeline`

        Return
        ------
        List of Evidence
        """
        pipeline_subset = self.get_pipeline(evaluator_name, metadata)
        pipeline_results = flatten_list(
            [step.evaluator.results for step in pipeline_subset]
        )
        evidences = []
        for result in pipeline_results:
            evidences += result.to_evidence()
        return evidences

    def get_pipeline(self, evaluator_name=None, metadata=None):
        """Returns pipeline or subset of pipeline steps

        Parameters
        ----------
        evaluator_name : str
            Name of evaluator to use to filter results. Must match the class name of an evaluator.
        metadata : dict
            Dictionary of evaluator metadata to filter results. Will return pipeline results
            whose metadata is a superset of the passed metadata

        Returns
        -------
        List of PipelineSteps
        """
        to_check = metadata or {}
        if evaluator_name:
            to_check["evaluator"] = evaluator_name
        return [p for p in self.pipeline if p.check_match(to_check)]

    def get_results(self, evaluator_name=None, metadata=None) -> List[Dict]:
        """
        Extract results from pipeline steps. Uses get_pipeline to determine to subset
        of pipeline steps to use

        Parameters
        ----------
        evaluator_name : str
            Name of evaluator to use to filter results. Must match the class name of an evaluator.
            Passed to `get_pipeline`
        metadata : dict
            Dictionary of evaluator metadata to filter results. Will return pipeline results
            whose metadata is a superset of the passed metadata. Passed to `get_pipeline`

        Returns
        -------
        Dict
            The format of the dictionary is Pipeline step id: results
        """
        pipeline_subset = self.get_pipeline(evaluator_name, metadata)
        pipeline_results = [
            {
                "metadata": step.metadata,
                "results": [r.df for r in step.evaluator.results],
            }
            for step in pipeline_subset
        ]
        return pipeline_results

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
        pipeline_step: PipelineStep,
        evaluator_arguments: dict,
    ):
        """
        Add a specific step while handling errors.

        Parameters
        ----------
        pipeline_step : PipelineStep
            An instance of a PipelineStep
        """
        pipeline_step.evaluator = pipeline_step.evaluator(**evaluator_arguments)
        ## Attempt pipe addition
        self.pipeline.append(pipeline_step)

        # Create logging message
        logger_message = f"Evaluator {pipeline_step.evaluator.name} added to pipeline. "
        metadata = pipeline_step.metadata
        if metadata is not None:
            if "dataset" in metadata:
                logger_message += f"Dataset used: {metadata['dataset']}. "
            if "sensitive_feature" in metadata:
                logger_message += f"Sensitive feature: {metadata['sensitive_feature']}"
        self.logger.info(logger_message)

    def _add_governance(self, governance: Governance = None):
        if governance is None:
            return
        self.gov = governance
        if self.model:
            self.gov.set_artifacts(self.model, self.training_data, self.assessment_data)

    def _cycle_add_through_ds_feat(
        self,
        pipeline_step,
        check_sens_feat,
        check_data,
        evaluator_arguments,
        features_to_eval,
    ):
        for feat in features_to_eval:
            additional_meta = {}
            if check_sens_feat:
                additional_meta["sensitive_feature"] = feat
            if check_data:
                for dataset_label, dataset in self.get_datasets().items():
                    additional_meta["dataset_type"] = dataset_label
                    step = deepcopy(pipeline_step)
                    step.metadata.update(additional_meta)
                    evaluator_arguments["data"] = dataset
                    self.change_sens_feat_view(evaluator_arguments, feat)
                    self._add(step, evaluator_arguments)
            else:
                self.change_sens_feat_view(evaluator_arguments, feat)
                step = deepcopy(pipeline_step)
                step.metadata.update(additional_meta)
                self._add(
                    step,
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
                if not pipeline:
                    self.logger.warning(
                        "No pipeline created from governance! Check that your"
                        " model is properly tagged. Try using Governance.tag_model"
                    )
            else:
                return
        # Create pipeline from list of steps
        for step in pipeline:
            if not isinstance(step, tuple):
                step = (step,)
            evaltr, meta = self._consume_pipeline_step(step)
            if isclass(evaltr):
                raise ValidationError(
                    f"Evaluator in step {step} needs to be instantiated"
                )
            self.add(evaltr, meta)
        return self

    def _validate(self):
        """
        Validate arguments passed to Lens. All checks should be here

        Raises
        ------
        ValidationError
        """
        if not (isinstance(self.assessment_data, Data) or self.assessment_data is None):
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

        if self.model is not None and self.gov is not None:
            if self.model.tags not in self.gov._unique_tags:
                mes = f"Model tags: {self.model.tags} are not among the once found in the governance object: {self.gov._unique_tags}"
                self.logger.warning(mes)

    @staticmethod
    def _consume_pipeline_step(step):
        def safe_get(step, index):
            return (step[index : index + 1] or [None])[0]

        evaltr = safe_get(step, 0)
        meta = safe_get(step, 1)
        return evaltr, meta

    @staticmethod
    def change_sens_feat_view(evaluator_arguments: Dict[str, Data], feat: str):
        for artifact in evaluator_arguments.values():
            if getattr(artifact, "active_sens_feat", False):
                artifact.active_sens_feat = feat
        return evaluator_arguments
