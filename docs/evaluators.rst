Evaluators
=======

Evaluators are the classes that perform specific functions on 
a model and/or data. These can include assessing the model for fairness, or profiling a 
data. Evaluators are constantly being added to the framework, which creates Lens's standard
library.

Library of Evaluators
-----------

DataEquity
    This evaluator assesses whether outcomes are distributed equally across a sensitive
    feature. Depending on the kind of outcome, different tests will be performed.

    - Discrete: chi-squared contingency tests,
      followed by bonferronni corrected posthoc chi-sq tests
    - Continuous: One-way ANOVA, followed by Tukey HSD posthoc tests
    - Proportion (Bounded [0-1] continuous outcome): outcome is transformed to logits, then
        proceed as normal for continuous

DataFairness
    This evaluator performs a fairness evaluation on the dataset. Given a sensitive feature,
    it calculates a number of assessments:

    - group differences of features
    - evaluates whether features in the dataset are proxies for the sensitive feature
    - whether the entire dataset can be seen as a proxy for the sensitive feature
      (i.e., the sensitive feature is "redundantly encoded")

DataProfiling
    This evaluator runs the pandas profiler on a data. Pandas profiler calculates a number
    of descriptive statistics about the data.

ModelFairness
    This evaluator calculates performance metrics disaggregated by a sensitive feature, as
    well as evaluating the parity of those metrics.

    Handles any metric that can be calculated on a set of ground truth labels and predictions,
    e.g., binary classification, multiclass classification, regression.

ModelEquity
    This evaluator assesses whether model outcomes (i.e., predictions) are distributed equally 
    across a sensitive feature. Depending on the kind of outcome, different tests will be performed.

    - Discrete: chi-squared contingency tests,
      followed by bonferronni corrected posthoc chi-sq tests
    - Continuous: One-way ANOVA, followed by Tukey HSD posthoc tests
    - Proportion (Bounded [0-1] continuous outcome): outcome is transformed to logits, then
        proceed as normal for continuous

Performance
    This evaluator calculates overall performance metrics.
    Handles any metric that can be calculated on a set of ground truth labels and predictions,
    e.g., binary classification, multiclass classification, regression.