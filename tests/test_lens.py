
import credoai.lens as cl
import numpy as np
import pandas as pd
from credoai.data._fetch_testdata import fetch_testdata
from sklearn.linear_model import LogisticRegression

# set up data and models
data = fetch_testdata()
X = data['data'][["experience"]]
y = data['target']
sensitive_feature = data['data']["gender"]

data = pd.concat([data['data'], y], axis=1)
credo_data = cl.CredoData(
    name="income_data", data=data, label_key='income', sensitive_feature_keys=['gender']
)
credo_training_data = cl.CredoData(
    name="income_data", data=data, label_key='income', sensitive_feature_keys=['gender']
)

gov = cl.CredoGovernance()
gov.model_id = 'model_test'
gov.use_case_id = 'use_case_test'
gov.dataset_id = 'dataset_test'

model = LogisticRegression(random_state=0).fit(X, y)
credo_model = cl.CredoModel(name="income_classifier", model=model)
assessment_plan = {"Fairness": {"metrics": ["precision_score"]},
                   "Performance": {"metrics": ["precision_score"]}}


def test_lens_with_model():
    lens = cl.Lens(model=credo_model, data=credo_data,
                   assessment_plan=assessment_plan)

    results = lens.run_assessments().get_results()
    metric = results["validation_model"]["Fairness"]["fairness"].index[0]
    score = round(results["validation_model"]["Fairness"]
                  ["fairness"].iloc[0]["value"], 2)
    expected_assessments = {'DatasetFairness', 'DatasetProfiling', 'DatasetEquity',
                            'ModelEquity', 'Fairness', 'Performance'}
    fairness_assessment = [record for record in lens.assessments if record.name ==
                           'validation_model'][0].assessments['Fairness']

    assert metric == "precision_score"
    assert score == 0.33
    assert set([a.name for a in lens.get_assessments(
        flatten=True)]) == expected_assessments
    assert fairness_assessment.initialized_module.metrics == [
        'precision_score']
