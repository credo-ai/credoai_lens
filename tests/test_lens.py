
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
    expected_assessments = {'DatasetFairness',
                            'DatasetProfiling', 'Fairness', 'Performance'}
    fairness_assessment = [record for record in lens.assessments if record.name ==
                           'validation_model'][0].assessments['Fairness']

    assert metric == "precision_score"
    assert score == 0.33
    assert set([a.name for a in lens.get_assessments(
        flatten=True)]) == expected_assessments
    assert fairness_assessment.initialized_module.metrics == [
        'precision_score']


def test_lens_without_model():
    lens = cl.Lens(data=credo_data)
    results = lens.run_assessments().get_results()
    metric_score = results["validation"]['DatasetFairness']["gender-demographic_parity_ratio"][0]['value']
    assert metric_score == 0.5
    assert set([a.name for a in lens.get_assessments(flatten=True)]) == {
        'DatasetFairness', 'DatasetProfiling'}


def test_lens_without_sensitive_feature():
    credo_data = cl.CredoData(
        name="income_data", data=data.drop('gender', axis=1), label_key='income'
    )
    lens = cl.Lens(model=credo_model, data=credo_data,
                   assessment_plan=assessment_plan)
    results = lens.run_assessments().get_results()
    expected_assessments = {'DatasetProfiling', 'Performance'}
    assert set([a.name for a in lens.get_assessments(
        flatten=True)]) == expected_assessments


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
    assert set([a.name for a in lens.get_assessments(flatten=True)]) == {
        'DatasetFairness', 'DatasetProfiling'}


def test_display():
    lens = cl.Lens(model=credo_model, data=credo_data,
                   assessment_plan=assessment_plan)
    lens.run_assessments()
    lens.display_results()


def test_asset_creation():
    lens = cl.Lens(model=credo_model, data=credo_data,
                   assessment_plan=assessment_plan, governance=gov)
    lens.run_assessments()
    lens.export('.')


def test_lens_with_model_and_training():
    lens = cl.Lens(model=credo_model, data=credo_data,
                   training_data=credo_training_data, assessment_plan=assessment_plan)

    results = lens.run_assessments().get_results()
    rule_based_attack_score = round(
        results["validation_training_model"]["Privacy"]["rule_based_attack_score"], 2)
    expected_assessments = {'DatasetFairness', 'DatasetProfiling',
                            'Fairness', 'Performance', 'Privacy', 'Security'}

    assert rule_based_attack_score == 0.5
    assert set([a.name for a in lens.get_assessments(
        flatten=True)]) == expected_assessments
