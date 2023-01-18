import requests
from credoai.utils import global_logger
from credoai._version import __version__


def validate_version():
    current_version = __version__

    package = "credoai-lens"  # replace with the package you want to check
    response = requests.get(f"https://pypi.org/pypi/{package}/json")
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
