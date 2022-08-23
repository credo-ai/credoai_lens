import credoai.lens as cl
import numpy as np
import pandas as pd
from credoai.data._fetch_testdata import fetch_testdata
from sklearn.linear_model import LogisticRegression

# set up data and models
train_data, test_data = fetch_testdata(False, 1, 1)

train_credo_data = cl.CredoData(
    name="test_data_train",
    X=train_data["X"],
    y=train_data["y"],
    sensitive_features=train_data["sensitive_features"],
)
test_credo_data = cl.CredoData(
    name="test_data_test",
    X=test_data["X"],
    y=test_data["y"],
    sensitive_features=test_data["sensitive_features"],
)

gov = cl.CredoGovernance()
gov.model_id = "model_test"
gov.use_case_id = "use_case_test"
gov.dataset_id = "dataset_test"

model = LogisticRegression(random_state=0).fit(train_data["X"], train_data["y"])
credo_model = cl.CredoModel(name="income_classifier", model=model)
assessment_plan = {
    "Fairness": {"metrics": ["precision_score"]},
    "Performance": {"metrics": ["precision_score"]},
}


def test_lens_with_model():
    lens = cl.Lens(
        model=credo_model, data=test_credo_data, assessment_plan=assessment_plan
    )

    results = lens.run_assessments().get_results()
    fairness_results = results["validation_model"]["Fairness"]["gender"]["fairness"]
    metric = fairness_results.index[0]
    metric_score = round(fairness_results.iloc[0]["value"], 2)
    expected_assessments = {
        "DatasetFairness",
        "DatasetProfiling",
        "DatasetEquity",
        "ModelEquity",
        "Fairness",
        "Performance",
    }
    fairness_assessment = [
        record for record in lens.assessments if record.name == "validation_model"
    ][0].assessments["Fairness"]

    assert metric == "precision_score"
    assert metric_score == 0.17
    assert (
        set([a.name for a in lens.get_assessments(flatten=True)])
        == expected_assessments
    )
    assert fairness_assessment.initialized_module.static_kwargs["metrics"] == [
        "precision_score"
    ]


def test_lens_without_model():
    lens = cl.Lens(data=test_credo_data)
    results = lens.run_assessments().get_results()
    metric_score = results["validation"]["DatasetFairness"]["gender"][
        "demographic_parity_ratio"
    ][0]["value"]
    assert round(metric_score, 2) == 0.8
    assert set([a.name for a in lens.get_assessments(flatten=True)]) == {
        "DatasetFairness",
        "DatasetProfiling",
        "DatasetEquity",
    }


def test_lens_without_sensitive_feature():
    test_credo_data = cl.CredoData(name="test_data", X=test_data["X"], y=test_data["y"])
    lens = cl.Lens(
        model=credo_model, data=test_credo_data, assessment_plan=assessment_plan
    )
    results = lens.run_assessments().get_results()
    expected_assessments = {"DatasetProfiling", "Performance"}
    assert (
        set([a.name for a in lens.get_assessments(flatten=True)])
        == expected_assessments
    )


def test_lens_with_intersectionality():
    test_credo_data = cl.CredoData(
        name="test_data",
        X=test_data["X"],
        y=test_data["y"],
        sensitive_features=test_data["sensitive_features"],
        sensitive_intersections=True,
    )
    lens = cl.Lens(
        model=credo_model, data=test_credo_data, assessment_plan=assessment_plan
    )
    results = lens.run_assessments().get_results()
    expected_sensitive_features = {"race", "gender", "race_gender"}
    assert (
        set(test_credo_data.sensitive_features.columns) == expected_sensitive_features
    )
    assert (
        set(results["validation"]["DatasetEquity"].keys())
        == expected_sensitive_features
    )
    expected_assessments = {
        "DatasetFairness",
        "DatasetProfiling",
        "DatasetEquity",
        "ModelEquity",
        "Fairness",
        "Performance",
    }
    assert (
        set([a.name for a in lens.get_assessments(flatten=True)])
        == expected_assessments
    )


def test_lens_dataset_with_missing_data():
    _, test_data = fetch_testdata(True, 1, 1)

    test_credo_data = cl.CredoData(
        name="test_data",
        X=test_data["X"],
        y=test_data["y"],
        sensitive_features=test_data["sensitive_features"],
    )

    lens = cl.Lens(
        data=test_credo_data,
        assessment_plan={"DatasetFairness": {"nan_strategy": "drop"}},
    )
    results = lens.run_assessments().get_results()

    metric_score = results["validation"]["DatasetFairness"]["gender"][
        "demographic_parity_ratio"
    ][0]["value"]
    assert round(metric_score, 2) == 0.44
    assert set([a.name for a in lens.get_assessments(flatten=True)]) == {
        "DatasetFairness",
        "DatasetProfiling",
        "DatasetEquity",
    }


def test_display():
    lens = cl.Lens(
        model=credo_model, data=test_credo_data, assessment_plan=assessment_plan
    )
    lens.run_assessments()
    lens.display_results()


def test_asset_creation():
    lens = cl.Lens(
        model=credo_model,
        data=test_credo_data,
        assessment_plan=assessment_plan,
        governance=gov,
    )
    lens.run_assessments()
    lens.export(".")


def test_lens_with_model_and_training():
    lens = cl.Lens(
        model=credo_model,
        data=test_credo_data,
        training_data=train_credo_data,
        assessment_plan=assessment_plan,
    )

    results = lens.run_assessments().get_results()
    rule_based_attack_score = round(
        results["validation_training_model"]["Privacy"][
            "MembershipInferenceBlackBoxRuleBased"
        ],
        2,
    )
    expected_assessments = {
        "DatasetFairness",
        "DatasetProfiling",
        "DatasetEquity",
        "ModelEquity",
        "Fairness",
        "Performance",
        "Privacy",
        "Security",
    }

    assert rule_based_attack_score == 0.42
    assert (
        set([a.name for a in lens.get_assessments(flatten=True)])
        == expected_assessments
    )
