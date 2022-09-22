import inspect
import sys


def list_assessments_exhaustive():
    """List all defined assessments"""
    return inspect.getmembers(
        sys.modules[__name__],
        lambda member: inspect.isclass(member) and member.__module__ == __name__,
    )


def list_assessments():
    """List subset of all defined assessments where the module is importable"""
    assessments = list_assessments_exhaustive()
    usable_assessments = []
    for assessment in assessments:
        try:
            _ = assessment[1]()
            usable_assessments.append(assessment)
        except AttributeError:
            pass
    return usable_assessments


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
