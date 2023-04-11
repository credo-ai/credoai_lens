<img src="https://raw.githubusercontent.com/credo-ai/credoai_lens/develop/docs/_static/images/credo_ai-lens.png" width="250" alt="Credo AI Lens"><br>

![Workflow](https://github.com/credo-ai/credoai_lens/actions/workflows/test-reports.yml/badge.svg)
![Tests](https://credoai-cicd-public-artifacts.s3.us-west-2.amazonaws.com/credoai_lens/main/tests-badge.svg)
[![Coverage](https://credoai-cicd-public-artifacts.s3.us-west-2.amazonaws.com/credoai_lens/main/coverage-badge.svg)](https://credoai-cicd-public-artifacts.s3.us-west-2.amazonaws.com/credoai_lens/main/html/index.html)

--------------------------------------

# Lens by Credo AI - Responsible AI Assessment Framework

Lens is a comprehensive assessment framework for AI systems. 
Lens standardizes model and data assessment, and acts as a central gateway to assessments 
created in the open source community. In short, Lens connects arbitrary AI models and datasets
with Responsible AI tools throughout the ecosystem.

Lens can be run in a notebook, a CI/CD pipeline, or anywhere else you do your ML analytics.
It is extensible, and easily customized to your organizations assessments if they are not 
supported by default. 

Though it can be used alone, Lens shows its full value when connected to your organization's 
[Credo AI App](https://www.credo.ai/product). Credo AI is an end-to-end AI Governance
App that supports multi-stakeholder alignment, AI assessment (via Lens) and AI risk assessment.



## Dependencies

- Credo AI Lens supports Python 3.8+
- Sphinx (optional for local docs site)


## Installation

The latest stable release (and required dependencies) can be installed from PyPI.

```
pip install credoai-lens
```

Additional installation instructions can be found in our [setup documentation](https://credoai-lens.readthedocs.io/en/stable/pages/setup.html)

## Getting Started
To get started, see the [quickstart demo](https://credoai-lens.readthedocs.io/en/stable/notebooks/quickstart.html).

If you are using the Credo AI Governance App, also check out the [governance integration demo](https://credoai-lens.readthedocs.io/en/stable/notebooks/governance_integration.html).

## Documentation

Documentation is hosted by [readthedocs](https://credoai-lens.readthedocs.io/en/stable/).

For dev documentation, see [latest](https://credoai-lens.readthedocs.io/en/stable/index.html).

## AI Governance

As an assessment framework, Lens is an important component of your overall **AI Governance** strategy.
But it's not the only component! Credo AI, the developer of Lens, also develops
tools to satisfy your general AI Governance needs, which integrate easily with Lens.

To connect to [Credo AI's Governance App](https://www.credo.ai/product), see the Governance
tutorial on [readthedocs](https://credoai-lens.readthedocs.io/en/stable/notebooks/governance_integration.html).
 
# For Lens developers

## Running tests


Running a test

```shell
scripts/test.sh
```

Running tests with pytest-watch

```shell
ptw --runner "pytest -s"
```
