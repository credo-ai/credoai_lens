"""
Tests integration with the platform.

Downloads an assessment plan, runs the assessments and exports
them to file and platform.

The test specific fixtures are leveraged by the `init_lens_integration`
fixture created in the tests.fixtures.lens_inits plugin.
"""
import pytest
import os
import base64
import json
from connect.governance.credo_api_client import CredoApiConfig


@pytest.fixture(scope="session")
def config_path_in():
    """
    Retrieves config path env variable.

    Not necessary in local mode as long as .config is in
    the expected location.
    """
    return os.getenv("CREDOAI_LENS_CONFIG_PATH", None)


@pytest.fixture(scope="session")
def api_config_in():
    """
    Retrieves config path env variable, in json format.

    Not necessary in local mode as long as .config is in
    the expected location.
    """
    b64_config = os.getenv("CREDOAI_LENS_CONFIG_JSON_B64", None)
    if b64_config:
        return CredoApiConfig(**json.loads(base64.b64decode(b64_config)))
    return None


@pytest.fixture(scope="session")
def assessment_plan_url_in():
    """
    Retrieves the assessment plan designated for this test.

    This is necessary also locally in order for the test to pass.
    """
    return os.getenv("CREDOAI_LENS_PLAN_URL", None)


def test_integration(init_lens_integration):
    lens, temp_file, gov = init_lens_integration

    lens.run()

    pytest.assume(lens.get_results())
    pytest.assume(lens.send_to_governance())
    pytest.assume(gov.export(temp_file))
    # Send results to Credo API Platform
    pytest.assume(gov.export())
