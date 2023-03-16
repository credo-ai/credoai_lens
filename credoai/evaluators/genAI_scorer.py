"""Requires installation of requirements-extras.txt"""

import os
from time import sleep

import numpy as np
import pandas as pd
from googleapiclient import discovery

from credoai.evaluators.evaluator import Evaluator
from credoai.utils import ValidationError

PERSPECTIVE_ATTRIBUTES = [
    "TOXICITY",
    "SEVERE_TOXICITY",
    "IDENTITY_ATTACK",
    "INSULT",
    "PROFANITY",
    "THREAT",
]


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
    assessment_functions_no_baseline : dict, optional
        Allows specification of arbitrary scoring functions that take as input a set of strings
         and return floating point scores. E.g., myScorer(responses) -> score
        Input dictionary should be of the form:
            {'callable': <scorer function object>, 'prepend_prompts': <Bool>, kwargs' : <arguments dict>}
        The callable-scorer pair is required and must take as input a list of strings and return a float.
        If prepend_prompts is true, prepend the prompts `self.data.X['prompts']` to the responses to be
        scored. If it is not provided, prompts are not prepended by default.
        The kwargs key is optional. It enables passing additional parameters to the scorer, such as
        API keys and rate limit parameters for API-based scorers.
    assessment_functions_w_baseline : dict, optional
        Allows specification of scoring functions that take as input a set of string prompts and a baseline
         dataset corresponding to expected outputs. Function must return a floating point score.
         E.g., myScorer(prompts, y) -> score
        Input dictionary should be of the form:
            {'callable': <scorer function object>, 'prepend_prompts': <Bool>, kwargs' : <arguments dict>}
        The callable-scorer pair is required and must take as input a list of strings and return a float.
        If prepend_prompts is true, prepend the prompts `self.data.X['prompts']` to the responses to be
        scored. If it is not provided, prompts are not prepended by default.
        The kwargs key is optional. It enables passing additional parameters to the scorer, such as
        API keys and rate limit parameters for API-based scorers.
    use_perspective_api : bool or list
        if True, use Perspective API to assess the generated responses. By default, LLM_Score will
        assess all possible perspective attributes. If a list of strings is passed, only those
        attributes will be assessed. Attributes can be selected from :attr:`PERSPECTIVE_ATTRIBUTES`.

        You must have a valid Perspective API key to use these functions defined as an enviornment
        variable called "PERSPECTIVE_API_KEY"
    perspective_rpm_limit : int, optional
        request per minute limit of your Perspective API account, by default 60
    """

    required_artifacts = {"model"}

    def __init__(
        self,
        assessment_functions_no_baseline={},
        assessment_functions_w_baseline={}
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
        if not isinstance(self.assessment_functions_no_baseline, dict):
            raise ValidationError(
                "'assessment_functions_no_baseline' values must be of type dict."
            )

        if not isinstance(self.assessment_functions_w_baseline, dict):
            raise ValidationError(
                "'assessment_functions_w_baseline' values must be of type dict."
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

        for name, function_spec in self.assessment_functions_no_baseline.items():
            if function_spec.get("prepend_prompts", False):
                self.generated_responses = self.data.X["prompt"].str.cat(
                    self.generated_responses
                )

            kwargs = function_spec.get("kwargs", {})
            scores[name] = function_spec["callable"](self.generated_responses, **kwargs)

        for name, function_spec in self.assessment_functions_w_baseline.items():
            if function_spec.get("prepend_prompts", False):
                self.generated_responses = self.data.X["prompt"].str.cat(
                    self.generated_responses
                )

            kwargs = function_spec.get("kwargs", {})
            scores[name] = function_spec["callable"](
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


def build_perspective_client(logger):
    """Build the self Perspective API client"""
    logger.info("LLM Score is building the Perspective API client")
    api_key = os.environ["PERSPECTIVE_API_KEY"]
    return discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=api_key,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        cache_discovery=False,
    )


def _assess_with_perspective(self):
    """Assess a text for a given assessment attribute

    Parameters
    ----------
    txt : str
        Text to be assessed

    Returns
    -------
    float
        assessment score
    """
    self.logger.info(
        "LLM Score is using the Perspective API to assess the generated responses."
    )
    if self.perspective_client is None:
        self._build_perspective_client()

    perspective_scores = []
    if self.use_perspective_api is True:
        assessment_attributes = PERSPECTIVE_ATTRIBUTES
    else:
        assessment_attributes = self.use_perspective_api
    completed_responses = self.data.X["prompt"].str.cat(self.generated_responses)

    for txt in completed_responses:
        analyze_request = {
            "comment": {"text": txt},
            "requestedAttributes": {att: {} for att in assessment_attributes},
            "languages": ["en"],
        }
        response = (
            self.perspective_client.comments().analyze(body=analyze_request).execute()
        )
        simplified_response = {
            f"perspective_{att}": response["attributeScores"][att]["summaryScore"][
                "value"
            ]
            for att in assessment_attributes
        }
        perspective_scores.append(simplified_response)
        sleep(60 / self.perspective_rpm_limit)
        return pd.DataFrame(perspective_scores).mean().to_dict()
