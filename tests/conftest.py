import os
import base64
import json

import pytest

from pytest import fixture

from connect.governance.credo_api_client import CredoApiConfig

pytest_plugins = [
    "tests.fixtures.datasets",
    "tests.fixtures.frozen_tests",
    "tests.fixtures.lens_artifacts",
    "tests.fixtures.lens_inits",
]


@pytest.fixture(scope="function")
def temp_file(tmp_path):
    """
    Creates a temporary file.

    This is used to test governance export to file.

    Parameters
    ----------
    tmp_path : pytest.fixture
        This is a pytest original fixture. It creates a temporary
        folder that gets auto removed once the test is complete.

    Returns
    -------
    Path
        Path to the temp file.
    """
    d = tmp_path / "test.json"
    d.touch()
    return d


########## Integration tests config ############


@fixture(scope="session")
def config_path_in():
    """
    Retrieves config path env variable.

    Not necessary in local mode as long as .config is in
    the expected location.
    """
    return os.getenv("CREDOAI_LENS_CONFIG_PATH", None)


@fixture(scope="session")
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


@fixture(scope="session")
def assessment_plan_url_in():
    """
    Retrieves the assessment plan designated for this test.

    This is necessary also locally in order for the test to pass.
    """
    return os.getenv("CREDOAI_LENS_PLAN_URL", None)
