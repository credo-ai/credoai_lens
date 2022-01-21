<img src="https://raw.githubusercontent.com/credo-ai/credoai_lens/develop/docs/_static/images/credo_ai-lens.png" width="250" alt="Credo AI Lens"><br>

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
[Credo AI Platform](https://www.credo.ai/product). Credo AI is an end-to-end AI Governance
platform that supports multi-stakeholder alignment, AI assessment (via Lens) and AI risk assesssment.



## Dependencies

- Credo AI Lens supports Python 3.7+
- Sphinx (optional for local docs site)


## Installation

The latest stable release (and required dependencies) can be installed from PyPI.
Note this installation only includes dependencies needed for a small set of modules

```
pip install credoai-lens
```

To include additional dependencies needed for some modules and demos, use the 
following installation command. 

```
pip install 'credoai-lens[extras]'
```

Modules that require extras include:
* fairness_nlp

### Failed pip installation - use conda

Installation sometimes fails on M1 macbooks. Specifically, pandas, scipy, and scikit-learn 
may fail to build. Installing these packages with the anaconda package manager seems to be
the easiest way to circumvent this issue. 

For development, the easiest way to interact with Lens is to use anaconda.

```
conda env create --file environment.yml
```

## Getting Started

To get started, see the [quickstart demo](https://credoai-lens.readthedocs.io/en/latest/notebooks/quickstart.html).

If you are using the Credo AI Governance Platform, also check out the [governance integration demo](https://credoai-lens.readthedocs.io/en/latest/notebooks/governance_integration.html).

## Documentation

Documentation is hosted by [readthedocs](https://credoai-lens.readthedocs.io/en/latest/index.html).

## Configuration

To connect to [Credo AI's Governance Platform](https://www.credo.ai/product), enter your connection info in `~/.credoconfig` (in the root directory) using
the below format. 

```
TENANT={tenant name} # Example: credoai
CREDO_URL=<your credo url>  # Example: https://api.credo.ai 
API_KEY=<your api key> # Example: JSMmd26...
```
 
