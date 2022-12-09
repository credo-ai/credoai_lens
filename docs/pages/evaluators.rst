Evaluators
==========

.. toctree::
   :maxdepth: 2
   :hidden:

   ./evaluators/dataprofiler
   ./evaluators/modelequity
   ./evaluators/modelfairness
   ./evaluators/performance
   ./evaluators/shapexplainer

   Experimental Evaluators <./evaluators/experimental>

Evaluators are the classes that perform specific functions on 
a model and/or data. These can include assessing the model for fairness, or profiling a 
data. Evaluators are constantly being added to the framework, which creates Lens's standard
library.

Library of Evaluators
---------------------

:ref:`DataProfiling<Data Profiler (Experimental)>`
    This evaluator runs the pandas profiler on a data. Pandas profiler calculates a number
    of descriptive statistics about the data.

:ref:`ModelEquity<Model Equity>`
    This evaluator assesses whether model outcomes (i.e., predictions) are distributed equally 
    across a sensitive feature. Depending on the kind of outcome, different tests will be performed.

    - Discrete: chi-squared contingency tests,
      followed by bonferronni corrected posthoc chi-sq tests
    - Continuous: One-way ANOVA, followed by Tukey HSD posthoc tests
    - Proportion (Bounded [0-1] continuous outcome): outcome is transformed to logits, then
        proceed as normal for continuous

:ref:`ModelFairness<Model Fairness>`
    This evaluator calculates performance metrics disaggregated by a sensitive feature, as
    well as evaluating the parity of those metrics.

    Handles any metric that can be calculated on a set of ground truth labels and predictions,
    e.g., binary classification, multiclass classification, regression.

:ref:`Performance`
    This evaluator calculates overall performance metrics.
    Handles any metric that can be calculated on a set of ground truth labels and predictions,
    e.g., binary classification, multiclass classification, regression.

