import os

from credoai.data._constants import _DOWNLOAD_DIRECTORY_NAME
from credoai.utils.common import get_project_root


def get_data_path(filename=None):
    data_dir = os.path.join(get_project_root(), "data", "static")
    if filename:
        data_dir = os.path.join(data_dir, filename)
    return data_dir


def get_local_credo_path(filename=None):
    data_dir = os.path.join(os.path.expanduser("~"), _DOWNLOAD_DIRECTORY_NAME)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if filename:
        data_dir = os.path.join(data_dir, filename)
    return data_dir
