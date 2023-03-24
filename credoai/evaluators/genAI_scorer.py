"""Requires installation of requirements-extras.txt"""

import numpy as np

from credoai.evaluators.evaluator import Evaluator
from credoai.utils import ValidationError


def assessment_function_no_baseline(callable, prepend_prompts=False, kwargs={}):
    # wonder if I can make a template like this work...
    pass


class LLM_Score(Evaluator):
    """
    This module assesses language generation models based on various prompts and assessment attributes.
    This evaluator generates responses using lens.data.X['prompts']. The responses are evaluated by the
    the specified assessment functions. Scorers should return a floating point number and can be either
    unary operators, accepting only the responses (e.g., myScorer(responses) -> score) or binary
    operators, accepting a benchmark for responses (e.g., myScorer(responses, y) -> score).

    Parameters
    ----------
    assessment_functions_no_baseline : List, optional
        Allows specification of arbitrary scoring functions that take as input a list of strings
         and return floating point scores. E.g., myScorer(responses) -> score
        Each entry in the input list should be a dictionary of the following form:
            {'name': <string>, 'callable': <scorer function object>, 'prepend_prompts': <Bool>, kwargs' : <arguments dict>}
        The name key and value are required.
        The callable-scorer function pair is required and must take as input a list of strings and return a float.
        If prepend_prompts is true, prepend the prompts `self.data.X['prompts']` to the responses to be
        scored. If it is not provided, prompts are not prepended by default.
        The kwargs key is optional. It enables passing additional parameters to the scorer, such as
        API keys and rate limit parameters for API-based scorers.
    assessment_functions_w_baseline : List, optional
        Allows specification of scoring functions that take as input a set of string prompts and a baseline
         dataset corresponding to expected outputs. Function must return a floating point score.
         E.g., myScorer(prompts, y) -> score
        Each entry in the input list should be a dictionary of the form:
            {'name': <string>, 'callable': <scorer function object>, 'prepend_prompts': <Bool>, kwargs' : <arguments dict>}
        The name key and value are required.
        The callable-scorer pair is required and must take as input a list of strings and return a float.
        If prepend_prompts is true, prepend the prompts `self.data.X['prompts']` to the responses to be
        scored. If it is not provided, prompts are not prepended by default.
        The kwargs key is optional. It enables passing additional parameters to the scorer, such as
        API keys and rate limit parameters for API-based scorers.
    """

    required_artifacts = {"model"}

    def __init__(
        self,
        assessment_functions_no_baseline=[],
        assessment_functions_w_baseline=[]
        # use_perspective_api=False,
        # perspective_rpm_limit=60,
    ):
        super().__init__()
        self.generated_responses = []
        self.assessment_functions_no_baseline = assessment_functions_no_baseline
        self.assessment_functions_w_baseline = assessment_functions_w_baseline
        # set up perspective api attributes
        # self.use_perspective_api = use_perspective_api
        # self.perspective_rpm_limit = perspective_rpm_limit
        # self.perspective_client = None

    def _validate_arguments(self):
        # TODO: validate model

        # TODO: check to make sure y is valid if baseline scorer used

        # type validation of scorers
        if not isinstance(self.assessment_functions_no_baseline, list):
            raise ValidationError(
                "'assessment_functions_no_baseline' values must be of type list."
            )

        if not isinstance(self.assessment_functions_w_baseline, list):
            raise ValidationError(
                "'assessment_functions_w_baseline' values must be of type list."
            )

    def _setup(self):
        # Perform prerun checks
        # TODO: uncomment and fix prerun checks
        # self._perform_prerun_checks()
        self.logger.info(
            "Performed prerun checks of generation and assessment functions"
        )

        return self

    def evaluate(self):
        """
        Run performance base module
        """
        self._generate_responses()

        scores = {}

        for function_spec in self.assessment_functions_no_baseline:
            if function_spec.get("prepend_prompts", False):
                self.generated_responses = self.data.X["prompt"].str.cat(
                    self.generated_responses
                )

            kwargs = function_spec.get("kwargs", {})
            scores[function_spec["name"]] = function_spec["callable"](
                self.generated_responses, **kwargs
            )

        for function_spec in self.assessment_functions_w_baseline:
            if function_spec.get("prepend_prompts", False):
                self.generated_responses = self.data.X["prompt"].str.cat(
                    self.generated_responses
                )

            kwargs = function_spec.get("kwargs", {})
            scores[function_spec["name"]] = function_spec["callable"](
                self.generated_responses, y=self.data.y, **kwargs
            )

        # TODO: convert scores into evidence objects
        return scores

    def _generate_responses(self):
        prompts = self.data.X["prompt"].tolist()
        self.generated_responses = self.model.generate(prompts)

    def _assess_against_y(self):
        """Assess the generated responses against target response

        Target responses are currently only used by anthropic-style evaluators, where
        the target responses determine whether the generated responses reflect the measured
        dimension or not.

        TODO: move from a simple string match to a more robust yes/no classifier
        """
        self.logger.info(
            "LLM Score is assessing the generated responses by comparing against y."
        )
        return np.mean(np.array(self.generated_responses) == self.data.y)

    def _perform_prerun_checks(self):
        """Checks the provided configurations and the generation and assessment functions

        Raises
        ------
        ValidationError
            Occurs if checks are not successfully completed
        """
        # Check the generation functions
        test_prompt = "To be, or not to be, that is"
        try:
            response = self.model.generate(test_prompt)
            if not isinstance(response, str):
                raise ValidationError(
                    self.model.name
                    + " failed to generate a string response for the test prompt '"
                    + test_prompt
                    + "'"
                )
        except:
            raise ValidationError(
                self.model.name
                + " failed to generate a response for the test prompt '"
                + test_prompt
                + "'"
            )

        # Check the assessment functions
        test_response = "The slings and arrows of outrageous fortune"
        for assessment_attribute, assessment_fun in self.assessment_functions.items():
            if assessment_fun in list(PERSPECTIVE_API_MODELS):
                if self.perspective_config is None:
                    raise ValidationError(
                        "Requested using '"
                        + assessment_fun
                        + "' but 'perspective_config' has not been provided to NLPGeneratorAnalyzer"
                    )
                for k in ["api_key", "rpm_limit"]:
                    if k not in self.perspective_config:
                        raise ValidationError(
                            "The provided 'perspective_config' is missing '" + k + "'"
                        )
                try:
                    self._assess_with_perspective(
                        test_response, PERSPECTIVE_API_MODELS[assessment_fun]
                    )
                except:
                    raise ValidationError(
                        "Perspective API function '"
                        + assessment_attribute
                        + "' failed to return a score for the test text '"
                        + test_response
                        + "'"
                    )
            else:
                try:
                    assessment_fun(test_response)
                except:
                    raise ValidationError(
                        "Assessment function '"
                        + assessment_attribute
                        + "' failed to return a score for the test text '"
                        + test_response
                        + "'"
                    )
