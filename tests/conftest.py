"""
Hosts generic fixtures that are not specific to datasets or
Lens artifacts.
"""
import pytest


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
