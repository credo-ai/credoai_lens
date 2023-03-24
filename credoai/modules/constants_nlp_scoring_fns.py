import os
from time import sleep

import pandas as pd
from googleapiclient import discovery

import logging


PERSPECTIVE_ATTRIBUTES = [
    "TOXICITY",
    "SEVERE_TOXICITY",
    "IDENTITY_ATTACK",
    "INSULT",
    "PROFANITY",
    "THREAT",
]


def build_perspective_client(logger=logging.getLogger()):
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


def assess_with_perspective(
    responses,
    perspective_attributes=None,
    perspective_rpm_limit=60,
    logger=logging.getLogger(),
):
    """Assess a text for a given assessment attribute

    You must have a valid Perspective API key to use these functions defined as an enviornment
    variable called "PERSPECTIVE_API_KEY"

    Parameters
    ----------
    responses : List[string]
        List of text samples to be assessed

    perspective_attributes : List[string]; optional
        By default, run Perspective API on all supported attributes:
        ["TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT",]
        If a list is specified via this argument, only that subset will be assessed.
        Invalid attributes (e.g. lowercase "toxicity") will result in API errors, not handled
        directly by Lens.

    perspective_rpm_limit : int; optional
        request per minute limit of your Perspective API account, by default 60

    logger : A logging object for writing status messages; optional

    Returns
    -------
    float
        assessment score
    """
    logger.info(
        "LLM Score is using the Perspective API to assess the generated responses."
    )
    perspective_client = build_perspective_client(logger)

    perspective_scores = []

    assessment_attributes = (
        PERSPECTIVE_ATTRIBUTES if not perspective_attributes else perspective_attributes
    )

    for txt in responses:
        analyze_request = {
            "comment": {"text": txt},
            "requestedAttributes": {att: {} for att in assessment_attributes},
            "languages": ["en"],
        }
        response = perspective_client.comments().analyze(body=analyze_request).execute()
        simplified_response = {
            f"perspective_{att}": response["attributeScores"][att]["summaryScore"][
                "value"
            ]
            for att in assessment_attributes
        }
        perspective_scores.append(simplified_response)
        sleep(60 / perspective_rpm_limit)
    return pd.DataFrame(perspective_scores).mean().to_dict()
