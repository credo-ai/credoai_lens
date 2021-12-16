#! /usr/bin/env python
#
# Copyright (C) 2021-2021 Credo AI
from credoai import __version__

DESCRIPTION = "credoai: AI governance tools"
LONG_DESCRIPTION = """\
Credo is a library that supports AI Governance using the Credo AI governance platform.
Here is some of the functionality that Credo offers:
- A set of integration tools to send data easily to Credo's governance platform
Credo aims to make interacting with Credo's governance platform as seamless as possible. 
"""

DISTNAME = 'credoai'
MAINTAINER = 'Ian Eisenberg'
MAINTAINER_EMAIL = 'ian@credo.com'
URL = ''
LICENSE = ''
DOWNLOAD_URL = 'https://github.com/Credo-AI/credo_toolkit'
VERSION = __version__
PYTHON_REQUIRES = ">=3.7"

# Use requirements.txt to set the install_requires
with open('requirements.txt') as f:
    INSTALL_REQUIRES = [line.strip() for line in f]

# Fetch extra requirements files
with open("requirements-extras.txt") as f:
    extras_requirements = [line.strip() for line in f]

EXTRAS_REQUIRES = {'extras' : extras_requirements}

PACKAGES = [
    'credoai',
    'credoai.utils',
    'credoai.reporting',
    'credoai.assessment',
    'credoai.modules',
    'credoai.modules.model_assessments',
    'credoai.modules.dataset_assessments'
]

CLASSIFIERS = [
    'Intended Audience :: Information Technology',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Operating System :: OS Independent',
]

PACKAGE_DATA = {
    'credoai':
        [
        'data/*',
        'data/nlp_generation_analyzer/persisted_models/*',
        'data/nlp_generation_analyzer/prompts/*'
        ]
}


if __name__ == "__main__":

    from setuptools import setup

    import sys
    if sys.version_info[:2] < (3, 6):
        raise RuntimeError("seaborn requires python >= 3.6.")

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRES,
        packages=PACKAGES,
        classifiers=CLASSIFIERS,
        include_package_data=True,
        package_data=PACKAGE_DATA
    )