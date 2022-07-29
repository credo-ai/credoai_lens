"""
CredoAssessments are adapters connecting CredoModels and/or CredoData
to Modules
"""

from .assessments import list_assessments
from .utils import *

usable_assessments = list_assessments()
globals().update({k: mod for k, mod in usable_assessments})
