
import numpy as np
import pandas as pd

import credoai.lens as cl
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

model = LogisticRegression(random_state=0).fit(X, y)
credo_model = cl.CredoModel(name="income_classifier", model=model)
assessment_plan = {"Fairness": {"metrics": ["precision_score"]},
                    "Performance": {"metrics": ["precision_score"]}}

def test_lens_with_model():
    lens = cl.Lens(model=credo_model, data=credo_data, assessment_plan=assessment_plan)

    results = lens.run_assessments().get_results()
    metric = results["validation"]["Fairness"]["fairness"].index[0]
    score = round(results["validation"]["Fairness"]["fairness"].iloc[0]["value"], 2)
    expected_assessments = {'DatasetFairness', 'DatasetProfiling', 'Fairness', 'Performance'}
    
    assert metric == "precision_score"
    assert score == 0.33
    assert set([a.name for a in lens.get_assessments(flatten=True)]) == expected_assessments 
    assert lens.assessments["validation"]['Fairness'].initialized_module.metrics == ['precision_score']

def test_lens_without_model():
    lens = cl.Lens(data=credo_data)
    results = lens.run_assessments().get_results()
    metric_score = results["validation"]['DatasetFairness']["gender-demographic_parity_ratio"][0]['value']
    assert metric_score == 0.5
    assert set([a.name for a in lens.get_assessments(flatten=True)]) == {'DatasetFairness', 'DatasetProfiling'} 

def test_lens_without_sensitive_feature():
    credo_data = cl.CredoData(
        name="income_data", data=data.drop('gender', axis=1), label_key='income'
    )
    lens = cl.Lens(model=credo_model, data=credo_data, spec=alignment_spec)
    results = lens.run_assessments().get_results()
    expected_assessments = {'DatasetProfiling', 'Performance'}
    assert set([a.name for a in lens.get_assessments(flatten=True)]) == expected_assessments 

def test_lens_dataset_with_missing_data():
    np.random.seed(0)
    data = fetch_testdata(add_nan=True)
    X = data['data'][["experience"]]
    y = data['target']
    sensitive_feature = data['data']["gender"]

    data = pd.concat([data['data'], y], axis=1)
    credo_data = cl.CredoData(
        name="income_data", data=data, label_key='income', sensitive_feature_keys=['gender']
    )

    lens = cl.Lens(data=credo_data)
    results = lens.run_assessments().get_results()
    metric_score = results["validation"]['DatasetFairness']["gender-demographic_parity_ratio"][0]['value']
    assert metric_score == 0.375
    assert set([a.name for a in lens.get_assessments(flatten=True)]) == {'DatasetFairness', 'DatasetProfiling'} 

def test_report_creation():
    lens = cl.Lens(model=credo_model, data=credo_data, assessment_plan=assessment_plan)
    lens.run_assessments()
    out = lens.create_report()