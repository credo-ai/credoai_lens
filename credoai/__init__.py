"""
Primary interface for Credo AI Lens package
"""

from importlib.metadata import version

# get package version programatically from installed wheel
# does not rely on update to __init__.py for each new release
# __version__ = version("credoai-lens")
__version__ = "1.0.0"


from credoai.utils.version_check import validate_version

validate_version()
