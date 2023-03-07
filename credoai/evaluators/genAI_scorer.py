"""Requires installation of requirements-extras.txt"""

import os
from functools import partial
from time import sleep

import numpy as np
import pandas as pd
from googleapiclient import discovery

from credoai.evaluators.evaluator import Evaluator
from credoai.utils import ValidationError

PERSPECTIVE_API_MODELS = {
    "perspective_toxicity": "TOXICITY",
    "perspective_severe_toxicity": "SEVERE_TOXICITY",
    "perspective_identify_attack": "IDENTITY_ATTACK",
    "perspective_insult": "INSULT",
    "perspective_profanity": "PROFANITY",
    "perspective_threat": "THREAT",
}


class LLM_Score(Evaluator):
    """
    This module assesses language generation models based on various prompts and assessment attributes

    Parameters
    ----------
    assessment_functions : dict
        keys are names of the assessment functions and values could be custom callable assessment functions
        or name of builtin assessment functions.

        Current choices for built-in assessment functions all use Perspective API include:
                'perspective_toxicity', 'perspective_severe_toxicity',
                'perspective_identify_attack', 'perspective_insult',
                'perspective_profanity', 'perspective_threat'

        You must have a valid Perspective API key to use these functions defined as an enviornment
        variable called "PERSPECTIVE_API_KEY"

    perspective_config : dict
        if Perspective API is to be used, this must be passed with the following:
            'api_key': your Perspective API key
            'rpm_limit': request per minute limit of your Perspective API account
    """

    required_artifacts = {"model"}

    def __init__(
        self,
        assessment_functions=None,
        perspective_config=None,
    ):
        super().__init__()
        self.assessment_functions = assessment_functions or {}
        self.perspective_config = perspective_config
        self.perspective_client = None
        self.generated_responses = []

    def _validate_arguments(self):
        # TODO: validate model

        # type validation of scorers
        if not isinstance(self.assessment_functions, dict):
            raise ValidationError("'assessment_functions' values must be of type dict.")

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

        # Generate and record responses for the prompts with all the generation models n_iterations times
        self._generate_responses()
        scores = {}
        if self.data.y:
            scores[self.data.name] = self._assess_against_y()
        for name, scorer in self.assessment_functions.items():
            scores[name] = scorer(self.generated_responses)

    def _generate_responses(self):
        prompts = self.data.X["prompt"].tolist()
        n = 20
        responses = []
        for i in range(0, len(prompts), n):
            responses += self.model.generate(prompts[i : i + n])
        self.generated_responses = responses

    def _assess_against_y(self):
        """Assess the generated responses against target response

        Target responses are currently only used by anthropic-style evaluators, where
        the target responses determine whether the generated responses reflect the measured
        dimension or not.

        TODO: move from a simple string match to a more robust yes/no classifier
        """
        return np.mean(np.array(self.generated_responses) == self.data.y)

    def _assess_with_perspective(self, assessment_attributes):
        """Assess a text for a given assessment attribute

        Parameters
        ----------
        txt : str
            Text to be assessed
        assessment_attribute : str
            Attribute to be do the assessment based on

        Returns
        -------
        float
            assessment score
        """
        if self.perspective_client is None:
            self._build_perspective_client()

        perspective_scores = []
        completed_responses = self.data.X["prompt"].str.cat(self.generated_responses)

        for txt in completed_responses:
            analyze_request = {
                "comment": {"text": txt},
                "requestedAttributes": {att: {} for att in assessment_attributes},
                "languages": ["en"],
            }
            response = (
                self.perspective_client.comments()
                .analyze(body=analyze_request)
                .execute()
            )
            simplified_response = {
                f"perspective_{att}": response["attributeScores"][att]["summaryScore"][
                    "value"
                ]
                for att in assessment_attributes
            }
            perspective_scores.append(simplified_response)
            return pd.DataFrame(perspective_scores).mean().to_dict()

    def _build_perspective_client(self):
        """Build the self Perspective API client"""
        api_key = os.environ["PERSPECTIVE_API_KEY"]
        if self.perspective_client is None:
            self.perspective_client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=api_key,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                cache_discovery=False,
            )

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
