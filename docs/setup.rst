Setup
======

Installation
-------------

The latest stable release or development release can be installed from PyPI.


**Pip**

Lens is on PyPi and can be installed using pip. The below will install the latest stable release
of the package.

::

   pip install credoai-lens

The latest development version can also be installed directly from github.

::

   pip install git+https://github.com/credo-ai/credoai_lens#develop

**Conda**

Lens is not on conda-forge. However, you can create a working
python environment using Anaconda by following the below steps. 

First, download this `environment.yml <https://raw.githubusercontent.com/credo-ai/credoai_lens/develop/environment.yml>`_ file.

Then run...

::

   conda env create --file {path-to-environment.yml}

The above will install the latest stable version of Lens. If you prefer to use
the latest development version, follow the above by installing directly from
github using pip (see below).

**ARM Macbook installation troubleshooting**

Pip installation sometimes fails on arm64 macbooks. Some packages that occasionally have issues are:
pandas, scipy, scikit-learn, tensorflow, transformers.
Installing these packages with the miniforge package manager seems to be
the easiest way to circumvent this issue. Ensure you are using miniforge rather than anaconda!
Only miniforge supports arm64 processors. E.g.:

::

   conda install -c conda-forge -c huggingface pandas scipy scikit-learn tensorflow transformers

.. warning::
   Tensorflow in particular is difficult to install on M1+ macbooks (macbooks using
   the arm64 processor). Full installation of lens (with extras) requires tensorflow. 
   Resources below should help you get your installation working.

If your mac has an arm64 processor, it is advisable to upgrade to Monterey or later (12.3+). Doing
so greatly simplifies tensorflow's installation. In this case you can install it directly from
conda running the following command:

::

   conda install -c conda-forge tensorflow

If you are running an earlier OS (e.g., BigSur), installation can be trickier. Below
are some resources that could help:

* Apple has `official guidance <https://github.com/apple/tensorflow_macos/issues/153>`_ here.

Other tutorials:

* https://www.mrdbourke.com/setup-apple-m1-pro-and-m1-max-for-machine-learning-and-data-science/
* https://betterdatascience.com/install-tensorflow-2-7-on-macbook-pro-m1-pro/