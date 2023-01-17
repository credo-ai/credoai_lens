"""
Tests the functionality in the quickstart notebook.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from credoai.artifacts import ClassificationModel, TabularData
from credoai.datasets import fetch_creditdefault
from credoai.evaluators import ModelFairness, Performance, Privacy
from credoai.lens import Lens


def setup_artifacts():
    data = fetch_creditdefault()
    df = data["data"]
    df["target"] = data["target"].astype(int)

    # fit model
    model = RandomForestClassifier(random_state=42)
    X = df.drop(columns=["SEX", "target"])
    y = df["target"]
    sensitive_features = df["SEX"]
    (
        X_train,
        X_test,
        y_train,
        y_test,
        sensitive_features_train,
        sensitive_features_test,
    ) = train_test_split(X, y, sensitive_features, random_state=42)
    model.fit(X_train, y_train)

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        sensitive_features_train,
        sensitive_features_test,
        model,
    )


class TestQuickstart:
    (
        X_train,
        y_train,
        X_test,
        y_test,
        sensitive_features_train,
        sensitive_features_test,
        model,
    ) = setup_artifacts()

    def test_run_all(self):
        credo_model = ClassificationModel(
            name="credit_default_classifier", model_like=self.model
        )
        credo_data = TabularData(
            name="UCI-credit-default",
            X=self.X_test,
            y=self.y_test,
            sensitive_features=self.sensitive_features_test,
        )

        # Initialization of the Lens object
        lens = Lens(model=credo_model, assessment_data=credo_data)

        # initialize the evaluator and add it to Lens
        metrics = ["precision_score", "recall_score", "equal_opportunity"]
        lens.add(ModelFairness(metrics=metrics))
        lens.add(Performance(metrics=metrics))

        # run Lens
        lens.run()

        assert lens.get_results()

        pipeline = [
            (ModelFairness(metrics)),
            (Performance(metrics)),
        ]
        lens = Lens(model=credo_model, assessment_data=credo_data, pipeline=pipeline)

        lens.run()

        assert lens.get_results()

        credo_model = ClassificationModel(
            name="credit_default_classifier", model_like=self.model
        )

        credo_data = TabularData(
            name="UCI-credit-default",
            X=self.X_test,
            y=self.y_test,
            sensitive_features=self.sensitive_features_test,
        )

        lens.add(Privacy())

        # This evaluator isn't actually run. I'm keeping it here though, since
        # it's in the quickstart notebook.

        lens = Lens(model=credo_model, assessment_data=credo_data)
        metrics = ["precision_score", "recall_score", "equal_opportunity"]
        lens.add(ModelFairness(metrics=metrics))
        lens.run()

        assert lens.get_results()
