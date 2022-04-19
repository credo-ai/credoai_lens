from absl import logging
from credoai.assessment import list_assessments

ASSESSMENTS = list_assessments()


def get_assessment_names():
    return {name: assessment_class().name
            for name, assessment_class in ASSESSMENTS}


def get_assessment_requirements():
    return {name: assessment_class().get_requirements()
            for name, assessment_class in ASSESSMENTS}


def get_usable_assessments(credo_model=None, credo_data=None, candidate_assessments=None):
    """Selects usable assessments based on model and data capabililties

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
        if assessment.check_requirements(credo_model, credo_data):
            assessments[assessment.get_name()] = assessment
        elif candidate_assessments is not None:
            logging.warning(f"Model or Data does not conform to {assessment.name} assessment's requirements.\nAssessment will not be run")
    return assessments
