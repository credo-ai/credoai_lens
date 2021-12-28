# Documentation & Website Generation

Building the docs requires additional dependencies listed in [`./requirements-dev.txt`](./requirements-dev.txt).

This directory contains the content relevant to documentation & website
generation using `sphinx`. The most important resource is `conf.py` which
includes settings and extensions that `sphinx` uses.

These docs make use of Sphinx's autosummary recursion.
To see how to configure Sphinx to do this, see this [Github repo README](https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion).



# Building the installable distribution
```
python setup.py sdist bdist_wheel
```

Follow instructions [here](https://packaging.python.org/en/latest/tutorials/packaging-projects/) to upload to PyPi.

To install from the test PyPi server (useful before full deployment) run:
```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple credoai-lens
```