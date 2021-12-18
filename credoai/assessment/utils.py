from credoai.assessment.assessments import list_usable_assessments

ASSESSMENTS = list_usable_assessments()

def get_assessment_names():
    return {name: assessment_class().name
            for name, assessment_class in ASSESSMENTS}


def get_assessment_requirements():
    return {name: assessment_class().get_requirements()
            for name, assessment_class in ASSESSMENTS}


def get_usable_assessments(credo_model, credo_data):
    assessments = []
    for name, assessment_class in ASSESSMENTS:
        assessment = assessment_class()
        if assessment.check_requirements(credo_model, credo_data):
            assessments.append(assessment)
    return assessments
