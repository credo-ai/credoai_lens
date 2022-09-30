"""
Boilerplate code for connecting Lens to Credo AI Platform
For more information check: https://credoai-lens.readthedocs.io/en/release-1.0.0/notebooks/governance_integration.html

Requirements
------------

    Platform
    --------
    1. A registered use case exists on the Platform
    2. A token and the relative .credoconfig file have been generated

    Local environment
    -----------------
    1. Credoai is correctly installed in the environment
    2. User has access to training and/or assessment data, and a fitted model
    3. Location of the .credoconfig file is the home directory: ~/

"""

########################################################################

"""
Example: Run assessment plan downloaded from Credo AI Platform

Assumptions
-----------
    1. Model is a classification model
    2. Data is in tabular form    
"""


from credoai.artifacts import ClassificationModel, TabularData
from credoai.governance import Governance
from credoai.lens import Lens

# Get your assessment plan
gov = Governance()
url = "<your assessment plan url>"
gov.register(assessment_plan_url=url)

# Create artifacts needed for Lens
# **Note**
# Different models or datasets may need different Model/Data Classes
# e.g., RegressionModel instead of ClassificationModel
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
# Under the hood, a pipeline of evaluators is created based
# on the governance requirements
lens = Lens(model=credo_model, assessment_data=credo_data, governance=gov)
lens.run()

# Add evidence and export to Credo AI Platform
lens.send_to_governance()
gov.export()
