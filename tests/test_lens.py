import credoai.lens as cl
import pandas as pd

from sklearn.linear_model import LogisticRegression


def test_lens():
    """test for for lens.Lens"""
    df = pd.DataFrame(
        {
            "gender": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "experience": [0, 0.1, 0.2, 0.4, 0.5, 0.6, 0, 0.1, 0.2, 0.4, 0.5, 0.6],
            "income": [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
        }
    )

    X = df[["experience"]]
    y = df["income"]
    sensitive_feature = df["gender"]

    model = LogisticRegression(random_state=0).fit(X, y)

    credo_model = cl.CredoModel(name="income_classifier", model=model)

    credo_data = cl.CredoData(
        name="income_data", X=X, y=y, sensitive_features=sensitive_feature
    )

    alignment_spec = {"FairnessBase": {"metrics": ["precision_score"]}}

    lens = cl.Lens(model=credo_model, data=credo_data, spec=alignment_spec)

    results = lens.run_assessments().get_results()

    metric = results["FairnessBase"]["fairness"].index[0]
    score = round(results["FairnessBase"]["fairness"].iloc[0]["value"], 2)

    assert metric == "precision_score"
    assert score == 0.33
