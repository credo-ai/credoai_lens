import credoai.artifacts as ca
import numpy as np
import pandas as pd

from credoai.lens import Lens
from credoai.data._fetch_testdata import fetch_testdata
from sklearn.linear_model import LogisticRegression

# set up data and models
data = fetch_testdata()
X = data['data'][["experience"]]
y = data['target']
sensitive_feature = data['data']["gender"]

data = pd.concat([data['data'], y], axis=1)
credo_data = ca.CredoData(
    name="income_data", data=data, label_key='income', sensitive_feature_key='gender'
)

model = LogisticRegression(random_state=0).fit(X, y)
credo_model = ca.CredoModel(name="income_classifier", model=model)
alignment_spec = {"FairnessBase": {"metrics": ["precision_score"]},
                    "PerformanceBase": {"metrics": ["precision_score"]}}

def test_lens_with_model():
    lens = Lens(model=credo_model, data=credo_data, spec=alignment_spec)

    results = lens.run_assessments().get_results()
    metric = results["FairnessBase"]["fairness"].index[0]
    score = round(results["FairnessBase"]["fairness"].iloc[0]["value"], 2)

    assert metric == "precision_score"
    assert score == 0.33
    assert set(lens.assessments.keys()) == {'DatasetFairness', 'FairnessBase', 'PerformanceBase'} 
    assert lens.assessments['FairnessBase'].initialized_module.metrics == ['precision_score']

def test_lens_without_model():
    lens = Lens(data=credo_data)
    results = lens.run_assessments().get_results()
    metric_score = results['DatasetFairness']["demographic_parity_ratio"][0]['value']
    assert metric_score == 0.5
    assert set(lens.assessments.keys()) == {'DatasetFairness'} 

def test_lens_dataset_with_missing_data():
    np.random.seed(0)
    data = fetch_testdata(add_nan=True)
    X = data['data'][["experience"]]
    y = data['target']
    sensitive_feature = data['data']["gender"]

    data = pd.concat([data['data'], y], axis=1)
    credo_data = ca.CredoData(
        name="income_data", data=data, label_key='income', sensitive_feature_key='gender'
    )

    lens = Lens(data=credo_data)
    results = lens.run_assessments().get_results()
    metric_score = results['DatasetFairness']["demographic_parity_ratio"][0]['value']
    assert metric_score == 0.375
    assert set(lens.assessments.keys()) == {'DatasetFairness'} 

def test_report_creation():
    lens = Lens(model=credo_model, data=credo_data, spec=alignment_spec)
    lens.run_assessments()
    out = lens.create_reports()