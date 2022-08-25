"""
Data, models and other static resources needed for CredoAI lens
"""

from ._fetch_censusincome import fetch_censusincome
from ._fetch_creditdefault import fetch_creditdefault

__all__ = ["fetch_creditdefault", "fetch_censusincome"]

try:
    from ._load_lr_toxicity import *

    __all__.append("load_lr_toxicity")
except ModuleNotFoundError:
    pass
