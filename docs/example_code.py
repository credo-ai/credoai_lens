"""
Boilerplate code for basic Lens functionality

Requirements
------------

    1. Credoai is correctly installed in the environment
    2. User has access to datasets/ fitted model`

Examples
--------
    1. Basic run without connection to Credo AI Platform
    For more information check: https://credoai-lens.readthedocs.io/en/release-1.0.0/notebooks/quickstart.html

    2. Run including connection to Credo AI Platform
    For more information check: https://credoai-lens.readthedocs.io/en/release-1.0.0/notebooks/governance_integration.html

Assumptions
-----------
    1. Model is a classification model
    2. Data is in tabular form

"""

########################################################################

"""
Example 1: Evaluate Model Fairness
"""

from credoai.lens import Lens
from credoai.artifacts import ClassificationModel, TabularData
from credoai.evaluators import ModelFairness

# Create artifacts needed for Lens
credo_model = ClassificationModel(
    name="your model name",
    model_like="<your model>",
)
credo_data = TabularData(
    name="your data name",
    X="<your X data object>",
    y="<your y data object>",
    sensitive_features="<your sensitive features object>",
)

# Initialization of the Lens object
lens = Lens(model=credo_model, assessment_data=credo_data)

# Initialize the evaluator and add it to Lens
metrics = ["precision_score", "recall_score", "equal_opportunity"]
lens.add(ModelFairness(metrics=metrics), id="MyModelFairness")

# run Lens
lens.run()
lens.get_results()["MyModelFairness"][0]

########################################################################

"""
Example 2: Run assessment plan downloaded from Credo AI Platform

Assumptions
-----------
    1. A registered use case exists on the Platform
    2. A token and the relative .credoconfig file have been generated
    3. Location of the config file is the home directory: ~/
"""


from credoai.governance import Governance
from credoai.lens import Lens
from credoai.artifacts import ClassificationModel, TabularData

# Get your assessment plan
gov = Governance()
url = "<your assessment plan url>"
gov.register(assessment_plan_url=url)

# Create artifacts needed for Lens
credo_model = ClassificationModel(
    name="your model name",
    model_like="<your model>",
)
credo_data = TabularData(
    name="your data name",
    X="<your X data object>",
    y="<your y data object>",
    sensitive_features="<your sensitive features object>",
)

# Pass governance to Lens at init
lens = Lens(model=credo_model, assessment_data=credo_data, governance=gov)
lens.run()

# Add evidence and export to Credo AI Platform
lens.send_to_governance(True)
gov.export()
