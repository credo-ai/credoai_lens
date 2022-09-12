# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import furo

sys.path.insert(0, os.path.abspath(".."))
import credoai

# -- Project information -----------------------------------------------------

project = "Credo AI | Lens"
copyright = "2021, Credo AI Development Team"
author = "Credo AI Development Team"
release = credoai.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx_copybutton",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",  # needs to be AFTER napoleon
    "sphinx_rtd_theme",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Auto-Doc Options
# ----------------
# Change the ordering of the member documentation
autodoc_member_order = "groupwise"
autoclass_content = "both"
autosummary_generate = True  # Turn on sphinx.ext.autosummary
# Enable 'expensive' imports for sphinx_autodoc_typehints
set_type_checking_flag = True
nbsphinx_allow_errors = True  # Continue through Jupyter errors
nbsphinx_execute = "never"  # do not execute jupyter notebooks

autodoc_mock_imports = [
    "art",
    "art.attacks.evasion",
    "art.attacks.extraction",
    "art.attacks.inference.membership_inference",
    "art.estimators.classification",
    "art.estimators.classification.scikitlearn",
    "cloudpickle",
    "dotenv",
    "fairlearn",
    "fairlearn.metrics",
    "googleapiclient",
    "ipywidgets",
    "joblib",
    "json_api_doc",
    "keras",
    "keras.layers",
    "keras.models",
    "keras.utils.np_utils",
    "matplotlib",
    "matplotlib.backends",
    "matplotlib.pyplot",
    "matplotlib.ticker",
    "nest_asyncio",
    "numpy",
    "pandas",
    "pandas_profiling",
    "scipy",
    "scipy.stats",
    "seaborn",
    "seaborn.utils",
    "sklearn",
    "sklearn.base",
    "sklearn.datasets",
    "sklearn.ensemble",
    "sklearn.metrics",
    "sklearn.preprocessing",
    "sklearn.utils",
    "sklearn.utils.multiclass",
    "sentence_transformers",
    "tempfile",
    "tensorflow",
    "transformers",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#


# documentation for furo: https://pradyunsg.me/furo/quickstart/

html_theme = "furo"

# import sphinx_rtd_theme
# html_theme = 'sphinx_rtd_theme'
# html_theme_options = {
#    'navigation_depth': 6
# }
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
