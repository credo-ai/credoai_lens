Experimental Evaluators
==========

.. toctree::
   :maxdepth: 2
   :hidden:

   ./experimental/dataequity
   ./experimental/datafairness
   ./experimental/deepchecks
   ./experimental/featuredrift
   ./experimental/identityverification
   ./experimental/modelprofiler
   ./experimental/privacy
   ./experimental/rankingfairness
   ./experimental/security
   ./experimental/survivalfairness

Several evaluators in Lens are still in an experimental development phase. The core functionality of these evaluators is stable. Nevertheless, users may occasionally encounter bugs and crashes. If you have feedback on these evaluators or encounter a bug, please engage with us on the project `Github repo <https://github.com/credo-ai/credoai_lens>`__.

:ref:`DataEquity<Data Equity (Experimental)>`
    This evaluator assesses whether outcomes are distributed equally across a sensitive
    feature. Depending on the kind of outcome, different tests will be performed.

    - Discrete: chi-squared contingency tests,
      followed by bonferronni corrected posthoc chi-sq tests
    - Continuous: One-way ANOVA, followed by Tukey HSD posthoc tests
    - Proportion (Bounded [0-1] continuous outcome): outcome is transformed to logits, then
        proceed as normal for continuous

:ref:`DataFairness<Data Fairness (Experimental)>`
    This evaluator performs a fairness evaluation on the dataset. Given a sensitive feature,
    it calculates a number of assessments:

    - group differences of features
    - evaluates whether features in the dataset are proxies for the sensitive feature
    - whether the entire dataset can be seen as a proxy for the sensitive feature
      (i.e., the sensitive feature is "redundantly encoded")

:ref:`RankingFairness<Ranking Fairness (Experimental)>`
    This evaluator calculates group fairness metrics for ranking systems.
    This works on ranked items. If items scores data are also available and provided, it outputs 
    a wider range of metrics.

:ref:`IdentityVerification<Identity Verification (Experimental)>`
    This evaluator performs performance and fairness assessments for identity verification systems.
    The identity verification system here refers to a pair-wise-comparison-based system that 
    inputs samples of a biometric attribute (face, fingerprint, voice, etc.) and their demographics
    and then outputs the degree to which they represent the same person to verify their Identity.

:ref:`Privacy<Privacy (Experimental)>`
    This evaluator calculates privacy metrics based on two adversarial attacks:

    - Membership inference attack: when an attacker with black-box access to a model attempts 
      to infer if a data sample was in the model's training dataset or not.
    - Attribute inference attack: when an attacker attempts to learn the attacked feature from 
      the rest of the features.

:ref:`Security<Security (Experimental)>`
    This evaluator calculates security metrics based on two adversarial attacks:

    - Model extraction attack: when an attacker with black-box access to a model attempts to 
      train a substitute model of it.
    - Model evasion attack: when an attacker with black-box access to a model attempts to
      create minimally-perturbed samples that get misclassified by the model.