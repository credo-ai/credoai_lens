
Model Equity
============


Evaluates the equity of a model's predictions.

This evaluator assesses whether model predictions are distributed equally across a sensitive
feature. Depending on the kind of outcome, different tests will be performed.

- Discrete: chi-squared contingency tests,
  followed by bonferronni corrected posthoc chi-sq tests
- Continuous: One-way ANOVA, followed by Tukey HSD posthoc tests
- Proportion (Bounded [0-1] continuous outcome): outcome is transformed to logits, then
    proceed as normal for continuous

Parameters
----------
use_predict_proba : bool, optional
    Defines which predict method will be used, if True predict_proba will be used.
    This methods outputs probabilities rather then class predictions. The availability
    of predict_proba is dependent on the model under assessment. By default False
p_value : float, optional
    The significance value to evaluate statistical tests, by default 0.01
