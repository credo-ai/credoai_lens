#! /usr/bin/env python
#
# Copyright (C) 2021-2021 Credo AI
import setuptools

# get version
exec(open("credoai/_version.py").read())
DESCRIPTION = "Lens: comprehensive assessment framework for AI systems"
DISTNAME = "credoai-lens"
MAINTAINER = "Ian Eisenberg"
MAINTAINER_EMAIL = "ian@credo.ai"
URL = ""
LICENSE = ""
DOWNLOAD_URL = "https://github.com/credo-ai/credoai_lens"
VERSION = __version__
PYTHON_REQUIRES = ">=3.7, <3.11"


# Fetch ReadMe
with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

# Use requirements.txt to set the install_requires
with open("requirements.txt") as f:
    INSTALL_REQUIRES = [line.strip() for line in f]

with open("requirements-extras.txt") as f:
    extras_requirements = [line.strip() for line in f]

# Fetch dev requirements files (including documentation)
with open("requirements-dev.txt") as f:
    dev_requirements = [line.strip() for line in f]

with open("docs/requirements.txt") as f:
    doc_requirements = [line.strip() for line in f]

dev_requirements += doc_requirements
full_requirements = extras_requirements

EXTRAS_REQUIRES = {"dev": dev_requirements, "full": full_requirements}


CLASSIFIERS = [
    "Intended Audience :: Information Technology",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Operating System :: OS Independent",
]

PACKAGE_DATA = {
    "credoai": [
        "datasets/*",
        "datasets/static/nlp_generator_analyzer/persisted_models/*",
        "datasets/static/nlp_generator_analyzer/prompts/*",
    ]
}


if __name__ == "__main__":
    import sys

    from setuptools import setup

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRES,
        packages=setuptools.find_packages(),
        classifiers=CLASSIFIERS,
        include_package_data=True,
        package_data=PACKAGE_DATA,
    )
