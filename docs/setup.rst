Setup
======

Installation
-------------

The latest stable release (and required dependencies) can be installed from PyPI.

.. note::
   This installation only includes dependencies needed for a small set of modules

``pip install credoai-lens``

.. note::
   To include additional dependencies needed for some modules and demos, use the 
   following installation command. 

On Mac
``pip install 'credoai-lens[extras]'``

On Windows
``pip install credoai-lens[extras]``


**Virtual Environment**

Lens can be installed in any python environment. The easiest way to create a working
python environment using Anaconda you can run the following command
before installing using pip.

``conda env create --file environment.yml``

The environment.yml file is found `here <https://github.com/credo-ai/credoai_lens/blob/develop/environment.yml>`_.

**ARM Macbook failed pip installation - use conda**

Installation sometimes fails on arm64 macbooks. Some packages that occasionally have issues are:
pandas, scipy, scikit-learn, tensorflow, transformers.
Installing these packages with the anaconda package manager seems to be
the easiest way to circumvent this issue. E.g.:

``conda install -c conda-forge -c huggingface pandas scipy scikit-learn tensorflow transformers``

.. warning::
   Tensorflow in particular is difficult to install on M1+ macbooks (macbooks using
   the arm64 processor). Full installation of lens (with extras) requires tensorflow. 
   Resources below should help you get your installation working.

If your mac has an arm64 processor, it is advisable to upgrade to Monterey or later (12.3+). Doing
so greatly simplifies tensorflow's installation. In this case you can install it directly from
conda running the following command:

``conda install -c conda-forge tensorflow``

If you are running an earlier OS (e.g., BigSur), installation can be trickier. Below
are some resources that could help:
* Apple has `official guidance <https://github.com/apple/tensorflow_macos/issues/153>`_ here.

Other tutorials:
* https://www.mrdbourke.com/setup-apple-m1-pro-and-m1-max-for-machine-learning-and-data-science/
* https://betterdatascience.com/install-tensorflow-2-7-on-macbook-pro-m1-pro/