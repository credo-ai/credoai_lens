"""
Main orchestration module handling running evaluators on AI artifacts
"""

from copy import deepcopy
from dataclasses import dataclass
from inspect import isclass
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from connect.governance import Governance
from joblib import Parallel, delayed

from credoai.artifacts import Data, Model
from credoai.evaluators.evaluator import Evaluator
from credoai.lens.pipeline_creator import PipelineCreator
from credoai.utils import (
    ValidationError,
    check_subset,
    flatten_list,
    global_logger,
)
from credoai.lens.lens_validation import check_model_data_consistency

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

    @property
    def id(self):
        eval_properties = self.evaluator.__dict__
        info_to_get = ["model", "assessment_data", "training_data", "data"]
        eval_info = pd.Series(
            [
                eval_properties.get(x).name if eval_properties.get(x) else "NA"
                for x in info_to_get
            ],
            index=info_to_get,
        )

        # Assign data to the correct dataset
        if eval_info.data != "NA":
            eval_info.loc[self.metadata["dataset_type"]] = eval_info.data
        eval_info = eval_info.drop("data").to_list()

        id = [self.metadata.get("evaluator", "NA")] + eval_info
        id.append(self.metadata.get("sensitive_feature", "NA"))
        return "~".join(id)

    def check_match(self, metadata):
        """
        Return true if metadata is a subset of pipeline step's metadata
        """
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
        n_jobs: int = 1,
    ):
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
        n_jobs : integer, optional
            Number of evaluator jobs to run in parallel.
            Uses joblib Parallel construct with multiprocessing backend.
            Specifying n_jobs = -1 will use all available processors.
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
        self.n_jobs = n_jobs
        self._add_governance(governance)
        self._validate()
        self._generate_pipeline(pipeline)
        # Can  pass pipeline directly

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

        # Run evaluators in parallel. Shared object (self.pipeline) necessitates writing
        # results to intermediate object evaluator_results
        evaluator_results = Parallel(n_jobs=self.n_jobs, verbose=100)(
            delayed(step.evaluator.evaluate)() for step in self.pipeline
        )

        # Write intermediate evaluator results back into self.pipeline for later processing
        for idx, evaluator in enumerate(evaluator_results):
            self.pipeline[idx].evaluator = evaluator
        return self

    def send_to_governance(self, overwrite_governance=True):
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
                "results": [r.data for r in step.evaluator.results],
            }
            for step in pipeline_subset
        ]
        return pipeline_results

    def print_results(self):
        results = self.get_results()
        for result_grouping in results:
            for key, val in result_grouping["metadata"].items():
                print(f"{key.capitalize()}: {val}")
            for val in result_grouping["results"]:
                print(f"{val}\n")
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
        artifact_args = {}
        if self.training_data:
            artifact_args["training_dataset"] = self.training_data.name
        if self.assessment_data:
            artifact_args["assessment_dataset"] = self.assessment_data.name
        if self.model:
            artifact_args["model"] = self.model.name
            artifact_args["model_tags"] = self.model.tags
            self.gov.set_artifacts(**artifact_args)

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
            evaltr, meta = self._consume_pipeline_step(deepcopy(step))
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

        # Validate combination of model and data
        if self.model is not None:
            for data_artifact in [self.assessment_data, self.training_data]:
                if data_artifact is not None:
                    check_model_data_consistency(self.model, data_artifact)

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
