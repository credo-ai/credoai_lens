from connect.governance import Governance
from credoai.lens import Lens
from credoai.artifacts import ClassificationModel, TabularData
from credoai.datasets import fetch_credit_model

import tempfile


def test_integration(config_path_in, assessment_plan_url_in):
    # Retrieve Policy Pack Assessment Plan
    gov = Governance(config_path=config_path_in)

    gov.register(assessment_plan_url=assessment_plan_url_in)

    (
        X_train,
        y_train,
        sensitive_features_train,
        X_test,
        y_test,
        sensitive_features_test,
        model,
    ) = fetch_credit_model(True)

    credo_model = ClassificationModel(
        name="credit_default_classifier",
        model_like=model,
        tags={"model_type": "binary_classification"},
    )

    credo_test = TabularData(
        name="UCI-credit-test",
        X=X_test,
        y=y_test,
        sensitive_features=sensitive_features_test,
    )

    credo_train = TabularData(
        name="UCI-credit-train",
        X=X_train,
        y=y_train,
        sensitive_features=sensitive_features_train,
    )

    # Initialization of the Lens object
    lens = Lens(
        model=credo_model,
        assessment_data=credo_test,
        training_data=credo_train,
        governance=gov,
    )

    lens.run()

    assert lens.get_results()

    lens.send_to_governance()
    # Send results to Credo API Platform

    tfile = tempfile.NamedTemporaryFile(delete=False)
    assert gov.export(tfile.name)

    assert gov.export()
