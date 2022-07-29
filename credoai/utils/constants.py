CREDO_URL = "https://api.credo.ai"
# used to determine when certain multiclass functionality should be executed
# e.g., label balance in dataset fairness
MULTICLASS_THRESH = 6

# mapping between risk issues and Lens modules
RISK_ISSUE_MAPPING = {
    "perf.1": "Performance",
    "fair.4": "Fairness",
    "fair.5": "Fairness",
    "fair.7": "Fairness",
}

MODEL_TYPES = ["CLASSIFIER", "REGRESSOR", "NEURAL_NETWORK"]

SUPPORTED_FRAMEWORKS = ("sklearn", "xgboost")
