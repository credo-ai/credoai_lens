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

Installation sometimes fails on M1 macbooks. Specifically, pandas, scipy, and scikit-learn 
may fail to build. Installing these packages with the anaconda package manager seems to be
the easiest way to circumvent this issue. 





Find more information about at `our github page <https://github.com/credo-ai/credoai_lens>`_.
