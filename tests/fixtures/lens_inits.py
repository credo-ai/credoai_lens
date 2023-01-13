"""
Contains all fixtures related to lens initialization.

The inits created here contain:

1. Initialized Lens instance
2. Governance instance
3. temp_file fixture (this is created in conftest.py)

Any fixture added to the file will be immediately available due to
addition of this module as plugin in the file `pytest.ini`.
"""
import pytest
from connect.governance import Governance
from pandas import DataFrame
from credoai.lens import Lens
from connect.governance.credo_api_client import CredoApiClient


@pytest.fixture(scope="function")
def init_lens_multiclass(
    multiclass_model, multiclass_assessment_data, multiclass_training_data, temp_file
):
    gov = Governance()
    my_pipeline = Lens(
        model=multiclass_model,
        assessment_data=multiclass_assessment_data,
        training_data=multiclass_training_data,
        governance=gov,
    )
    return my_pipeline, temp_file, gov


@pytest.fixture(scope="function")
def init_lens_classification(
    classification_model,
    classification_assessment_data,
    classification_train_data,
    temp_file,
):
    gov = Governance()
    my_pipeline = Lens(
        model=classification_model,
        assessment_data=classification_assessment_data,
        training_data=classification_train_data,
        governance=gov,
    )
    return my_pipeline, temp_file, gov


@pytest.fixture(scope="function")
def init_lens_credit(
    credit_classification_model,
    credit_assessment_data,
    credit_training_data,
    temp_file,
):
    gov = Governance()
    my_pipeline = Lens(
        model=credit_classification_model,
        assessment_data=credit_assessment_data,
        training_data=credit_training_data,
        governance=gov,
    )
    return my_pipeline, temp_file, gov


@pytest.fixture(scope="function")
def init_lens_integration(
    api_config_in,
    config_path_in,
    assessment_plan_url_in,
    temp_file,
    credit_classification_model_integration,
    credit_assessment_data,
    credit_training_data,
):
    # Retrieve Policy Pack Assessment Plan
    if api_config_in:
        gov = Governance(credo_api_client=CredoApiClient(config=api_config_in))
    else:
        gov = Governance(config_path=config_path_in)

    gov.register(assessment_plan_url=assessment_plan_url_in)

    # Initialization of the Lens object
    lens = Lens(
        model=credit_classification_model_integration,
        assessment_data=credit_assessment_data,
        training_data=credit_training_data,
        governance=gov,
    )
    return lens, temp_file, gov


@pytest.fixture(scope="function")
def init_lens_fairness(
    ranking_fairness_assessment_data,
    temp_file,
):
    expected_results = DataFrame(
        {
            "value": [0.11, 0.90, 0.20, 0.90, 0.67, 0.98, 0.65, 0.59],
            "type": [
                "skew_parity_difference",
                "skew_parity_ratio",
                "ndkl",
                "demographic_parity_ratio",
                "balance_ratio",
                "score_parity_ratio",
                "score_balance_ratio",
                "relevance_parity_ratio",
            ],
            "subtype": ["score"] * 8,
        }
    )
    gov = Governance()
    pipeline = Lens(assessment_data=ranking_fairness_assessment_data, governance=gov)

    return pipeline, temp_file, gov, expected_results


@pytest.fixture(scope="function")
def init_lens_identityverification(
    identity_verification_model,
    identity_verification_comparison_data,
    temp_file,
):
    expected_results_perf = DataFrame(
        {
            "value": [0.33, 1.00],
            "type": ["false_match_rate", "false_non_match_rate"],
            "subtype": ["score"] * 2,
        }
    )

    expected_results_fair = DataFrame(
        {
            "gender": ["female", "male", "female", "male"],
            "type": [
                "false_match_rate",
                "false_match_rate",
                "false_non_match_rate",
                "false_non_match_rate",
            ],
            "value": [0, 0, 0, 1],
        }
    )
    expected_results = {"fair": expected_results_fair, "perf": expected_results_perf}

    gov = Governance()
    pipeline = Lens(
        model=identity_verification_model,
        assessment_data=identity_verification_comparison_data,
        governance=gov,
    )

    return pipeline, temp_file, gov, expected_results


@pytest.fixture(scope="function")
def init_lens_regression(
    regression_model,
    regression_assessment_data,
    regression_train_data,
    temp_file,
):
    gov = Governance()
    my_pipeline = Lens(
        model=regression_model,
        assessment_data=regression_assessment_data,
        training_data=regression_train_data,
        governance=gov,
    )

    return my_pipeline, temp_file, gov
