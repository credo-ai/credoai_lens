
Data Equity
===========


Data Equity evaluator for Credo AI.

This evaluator assesses whether outcomes are distributed equally across a sensitive
feature. Depending on the kind of outcome, different tests will be performed.

- Discrete: chi-squared contingency tests,
  followed by bonferronni corrected posthoc chi-sq tests
- Continuous: One-way ANOVA, followed by Tukey HSD posthoc tests
- Proportion (Bounded [0-1] continuous outcome): outcome is transformed to logits, then
  proceed as normal for continuous

Parameters
----------
p_value : float
    The significance value to evaluate statistical tests
