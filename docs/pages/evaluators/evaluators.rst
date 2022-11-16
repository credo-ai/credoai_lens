Evaluators
==========

.. toctree::
   :maxdepth: 1
   :hidden:

   ./privacy
   ./make_your_own

Evaluators are the classes that perform specific functions on 
a model and/or data. These can include assessing the model for fairness, or profiling a 
data. Evaluators are constantly being added to the framework, which creates Lens's standard
library.

Library of Evaluators
---------------------

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

RankingFairness
    This evaluator calculates group fairness metrics for ranking systems.
    This works on ranked items. If items scores data are also available and provided, it outputs 
    a wider range of metrics.

IdentityVerification
    This evaluator performs performance and fairness assessments for identity verification systems.
    The identity verification system here refers to a pair-wise-comparison-based system that 
    inputs samples of a biometric attribute (face, fingerprint, voice, etc.) and their demographics
    and then outputs the degree to which they represent the same person to verify their Identity.

Privacy
    This evaluator calculates privacy metrics based on two adversarial attacks:

    - Membership inference attack: when an attacker with black-box access to a model attempts 
      to infer if a data sample was in the model's training dataset or not.
    - Attribute inference attack: when an attacker attempts to learn the attacked feature from 
      the rest of the features.

Security
    This evaluator calculates security metrics based on two adversarial attacks:

    - Model extraction attack: when an attacker with black-box access to a model attempts to 
      train a substitute model of it.
    - Model evasion attack: when an attacker with black-box access to a model attempts to
      create minimally-perturbed samples that get misclassified by the model.

