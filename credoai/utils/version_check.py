import requests

from credoai._version import __version__
from credoai.utils import global_logger


def validate_version():
    current_version = __version__

    package = "credoai-lens"  # replace with the package you want to check
    try:
        response = requests.get(f"https://pypi.org/pypi/{package}/json")
    except requests.ConnectionError:
        global_logger.info(
            "No internet connection. Cannot determine whether Credo AI Lens version is up-to-date"
        )
        return
    latest_version = response.json()["info"]["version"]

    on_latest = current_version == latest_version

    if not on_latest:
        global_logger.warning(
            """
            You are using credoai-lens version %s, however a newer version is available.
            Lens is updated regularly with major improvements and bug fixes.
            Please upgrade via the command: "python -m pip install --upgrade credoai-lens"
            """,
            current_version,
        )
