import credoai.lens as cl
import numpy as np
import pandas as pd
from credoai.data._fetch_testdata import fetch_testdata
from sklearn.linear_model import LogisticRegression

# set up data and models
train_data, test_data = fetch_testdata()

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


def test_lens_reliability():
    results = pd.DataFrame()
    for run_i in range(5):
        lens = cl.Lens(
            model=credo_model,
            data=test_credo_data,
            training_data=train_credo_data,
            assessment_plan=assessment_plan,
        ).run_assessments()

        run_results = pd.concat(
            [lens._prepare_results(a) for a in lens.get_assessments(True)]
        ).loc[:, ["value", "assessment", "metric_key"]]
        run_results["run"] = run_i
        results = pd.concat([results, run_results])
    results.reset_index(inplace=True)
    reliability_df = results.groupby(["metric_key", "assessment"]).value.agg(
        ["mean", "std"]
    )
    (reliability_df["std"] / reliability_df["mean"]).sort_values()
