from dataclasses import dataclass

from absl import logging
from credoai.assessment import list_assessments

ASSESSMENTS = list_assessments()


def get_assessment_names():
    return {name: assessment_class().name for name, assessment_class in ASSESSMENTS}


def get_assessment_requirements():
    return {
        name: assessment_class().get_requirements()
        for name, assessment_class in ASSESSMENTS
    }


def get_usable_assessments(
    credo_model=None,
    credo_data=None,
    credo_training_data=None,
    candidate_assessments=None,
):
    """Selects usable assessments based on model and data capabilities

    Parameters
    ----------
    credo_model : CredoModel
    credo_data : CredoData
    candidate_assessments : list, optional
        list of assessments to use. If None, search over all
        possible assessments, by default None

    Returns
    -------
    dict
        dictionary of assessments, {name: assessment, ...}
    """
    if candidate_assessments is None:
        to_check = [a[1] for a in ASSESSMENTS]
    else:
        to_check = candidate_assessments
    assessments = {}
    for assessment_class in to_check:
        assessment = assessment_class()
        if assessment.check_requirements(credo_model, credo_data, credo_training_data):
            assessments[assessment.get_name()] = assessment
    return assessments


@dataclass
class AssessmentBunch:
    """
    Class to determine assessment eligibility and hold a collections of assessments
    """

    name: str
    model: "CredoModel"
    primary_dataset: "CredoData"
    secondary_dataset: "CredoData"
    assessments: list = None

    def set_usable_assessments(self, candidate_assessments):
        self.assessments = get_usable_assessments(
            self.model,
            self.primary_dataset,
            self.secondary_dataset,
            candidate_assessments,
        )
