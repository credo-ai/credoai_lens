# Documentation & Website Generation

To build the documentation locally, run `make html` from the `/docs` directory and the docs site will build to: `docs/_build/html/index.html`, which can be opened in the browser.
> Make sure you have [Sphinx installed](https://www.sphinx-doc.org/en/master/usage/installation.html) if you are building the docs site locally.

Building the docs requires additional dependencies listed in `docs/requirements.txt`.

This directory contains the content relevant to documentation & website
generation using `sphinx`. The most important resource is `conf.py` which
includes settings and extensions that `sphinx` uses.

These docs make use of Sphinx's autosummary recursion.
To see how to configure Sphinx to do this, see this [Github repo README](https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion).



# Building the installable distribution
```
python setup.py sdist bdist_wheel
```

# Testing Package
## Upload to TestPyPI
After creating the required files in `dist` and installing twine, run:
```
python -m twine upload --repository testpypi dist/*
```
For more info, follow instructions [here](https://packaging.python.org/en/stable/tutorials/packaging-projects/) to upload to PyPi.

## Installating from test server
To install from the test PyPi server (useful before full deployment) run:
```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple credoai-lens
```

# Upload to Pypi
```
python -m twine upload dist/*
```