Metrics new auto version
========================

Lens supports many metrics out-of-the-box. 
The following gives a comprehensive list, which you can also generate in your python environment:

Below we provide details for a selection of these supported metrics. 

Custom metrics are supported by using the `Metric` class, which can be used to wrap any assessment function.


.. list-table:: List of all metrics
	:header-rows: 1

	* - Metric Name
	  - rai_dimension
	  - synonyms
	* - :ref:`accuracy_score<accuracy_score>`
	  - performance
	  - 
	* - :ref:`average_precision_score<average_precision_score>`
	  - performance
	  - average_precision
	* - :ref:`balanced_accuracy_score<balanced_accuracy_score>`
	  - performance
	  - 
	* - :ref:`d2_tweedie_score<d2_tweedie_score>`
	  - performance
	  - 
	* - :ref:`demographic_parity_difference<demographic_parity_difference>`
	  - fairness
	  - statistical_parity, demographic_parity
	* - :ref:`demographic_parity_ratio<demographic_parity_ratio>`
	  - fairness
	  - disparate_impact
	* - :ref:`det_curve<det_curve>`
	  - 
	  - detection_error_tradeoff
	* - :ref:`equal_opportunity_difference<equal_opportunity_difference>`
	  - fairness
	  - equal_opportunity
	* - :ref:`equalized_odds_difference<equalized_odds_difference>`
	  - fairness
	  - equalized_odds
	* - :ref:`evasion_attack_score<evasion_attack_score>`
	  - security
	  - 
	* - :ref:`explained_variance_score<explained_variance_score>`
	  - performance
	  - 
	* - :ref:`extraction_attack_score<extraction_attack_score>`
	  - security
	  - 
	* - :ref:`f1_score<f1_score>`
	  - performance
	  - 
	* - :ref:`false_discovery_rate<false_discovery_rate>`
	  - performance
	  - fdr
	* - :ref:`false_negative_rate<false_negative_rate>`
	  - performance
	  - miss_rate, fnr, false_non_match_rate
	* - :ref:`false_omission_rate<false_omission_rate>`
	  - performance
	  - 
	* - :ref:`false_positive_rate<false_positive_rate>`
	  - performance
	  - fpr, false_match_rate, fallout_rate
	* - :ref:`gain_chart<gain_chart>`
	  - 
	  - 
	* - :ref:`gini_coefficient<gini_coefficient>`
	  - 
	  - discriminatory_gini, discriminatory_gini_index, gini_index
	* - :ref:`ks_score_binary<ks_score_binary>`
	  - 
	  - ks_score
	* - :ref:`matthews_correlation_coefficient<matthews_correlation_coefficient>`
	  - performance
	  - 
	* - :ref:`max_error<max_error>`
	  - performance
	  - 
	* - :ref:`max_proxy_mutual_information<max_proxy_mutual_information>`
	  - 
	  - 
	* - :ref:`mean_absolute_error<mean_absolute_error>`
	  - performance
	  - MAE
	* - :ref:`mean_absolute_percentage_error<mean_absolute_percentage_error>`
	  - performance
	  - 
	* - :ref:`mean_gamma_deviance<mean_gamma_deviance>`
	  - performance
	  - 
	* - :ref:`mean_pinball_loss<mean_pinball_loss>`
	  - performance
	  - 
	* - :ref:`mean_poisson_deviance<mean_poisson_deviance>`
	  - performance
	  - 
	* - :ref:`mean_squared_error<mean_squared_error>`
	  - performance
	  - MSD, mean_squared_deviation, MSE
	* - :ref:`mean_squared_log_error<mean_squared_log_error>`
	  - performance
	  - 
	* - :ref:`median_absolute_error<median_absolute_error>`
	  - performance
	  - 
	* - :ref:`membership_inference_attack_score<membership_inference_attack_score>`
	  - 
	  - 
	* - :ref:`model_based_attack_score<model_based_attack_score>`
	  - 
	  - 
	* - :ref:`overprediction<overprediction>`
	  - performance
	  - 
	* - :ref:`precision_recall_curve<precision_recall_curve>`
	  - 
	  - pr_curve
	* - :ref:`precision_score<precision_score>`
	  - performance
	  - precision
	* - :ref:`r2_score<r2_score>`
	  - performance
	  - r2, r_squared
	* - :ref:`roc_auc_score<roc_auc_score>`
	  - performance
	  - 
	* - :ref:`roc_curve<roc_curve>`
	  - 
	  - 
	* - :ref:`root_mean_squared_error<root_mean_squared_error>`
	  - performance
	  - RMSE
	* - :ref:`rule_based_attack_score<rule_based_attack_score>`
	  - 
	  - 
	* - :ref:`selection_rate<selection_rate>`
	  - performance
	  - 
	* - :ref:`sensitive_feature_prediction_score<sensitive_feature_prediction_score>`
	  - performance
	  - 
	* - :ref:`target_ks_statistic<target_ks_statistic>`
	  - performance
	  - ks_score_regression, ks_score
	* - :ref:`true_negative_rate<true_negative_rate>`
	  - performance
	  - specificity, tnr
	* - :ref:`true_positive_rate<true_positive_rate>`
	  - performance
	  - recall_score, sensitivity, recall, tpr, hit_rate
	* - :ref:`underprediction<underprediction>`
	  - performance
	  - 

Accuracy_Score
--------------

Accuracy is the fraction of predictions that a classification model got right. This metric is not robust to class imbalance.

The best value is 1 and the worst value is 0.

.. math::

   \text{Accuracy} = \frac{Correct \ Predictions}{Total \ Number \ of \ Examples}

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html>`__

Average_Precision_Score
-----------------------

Average precision summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html>`__

**Other known names**: average_precision

Balanced_Accuracy_Score
-----------------------

The balanced accuracy in classification problems is used to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.

The best value is 1 and the worst value is 0.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html>`__

D2_Tweedie_Score
----------------

:math:`D^2`  regression score is percentage of `Tweedie deviance <https://en.wikipedia.org/wiki/Tweedie_distribution#The_Tweedie_deviance>`__ explained.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.d2_tweedie_score.html>`__

Demographic_Parity_Difference
-----------------------------

Demographic parity difference is a parity metric that is satisfied if the results of a model's classification are not dependent on a given sensitive attribute.

Demographic parity difference should be ideally 0.

**Source**: `click here <https://fairlearn.org/v0.4.6/api_reference/fairlearn.metrics.html#fairlearn.metrics.demographic_parity_difference>`__

**Other known names**: statistical_parity, demographic_parity

Demographic_Parity_Ratio
------------------------

Demographic parity ratio is a parity metric that is satisfied if the results of a model's classification are not dependent on a given sensitive attribute.

Demographic parity ratio should be ideally 1.

**Source**: `click here <https://fairlearn.org/v0.4.6/api_reference/fairlearn.metrics.html#fairlearn.metrics.demographic_parity_ratio>`__

**Other known names**: disparate_impact

Det_Curve
---------



**Other known names**: detection_error_tradeoff

Equal_Opportunity_Difference
----------------------------

The equalized odds difference is equivalent to the `true_positive_rate_difference` defined as the difference between the largest and smallest of :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a` of the sensitive feature(s).

**Source**: `click here <https://github.com/credo-ai/credoai_lens/blob/develop/credoai/metrics/metrics_credoai.py>`__

**Other known names**: equal_opportunity

Equalized_Odds_Difference
-------------------------

The equalized odds difference of 0 means that all groups have the same true positive, true negative, false positive, and false negative rates.

**Source**: `click here <https://fairlearn.org/v0.4.6/api_reference/fairlearn.metrics.html#fairlearn.metrics.equalized_odds_difference>`__

**Other known names**: equalized_odds

Evasion_Attack_Score
--------------------

Model evasion attack occurs when an attacker with black-box access to a model attempts to create minimally-perturbed samples that get misclassified by the model.

Model evasion attack score the success rate of this attack.

The best value is 0 and the worst value is 1.

**Source**: `click here <https://github.com/credo-ai/credoai_lens/blob/main/credoai/evaluators/security.py>`__

Explained_Variance_Score
------------------------

Explained variance regression score function.

Best possible score is 1.0, lower values are worse.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html>`__

Extraction_Attack_Score
-----------------------

Model extraction attack occurs when an attacker with black-box access to a model attempts to train a substitute model of it.

Model extraction attack score is the accuracy of the thieved model divided by the accuracy of the victim model, corrected for chance.

The best value is 0 and the worst value is 1.

**Source**: `click here <https://github.com/credo-ai/credoai_lens/blob/main/credoai/evaluators/security.py>`__

F1_Score
--------

Also known as balanced F-score or F-measure, the F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.

.. math::

   \text{False Positive Rate} = \frac{2 \times Precision \times Recall}{Precision + Recall}

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>`__

False_Discovery_Rate
--------------------

False discovery rate is intuitively the rate at which the classifier will be wrong when labeling an example as positive.

The best value is 0 and the worst value is 1.

.. math::

   \text{False Discovery Rate} = \frac{False \ Positives}{False \ Positives + True \ Positives}

**Source**: `click here <https://github.com/credo-ai/credoai_lens/blob/develop/credoai/metrics/metrics_credoai.py>`__

**Other known names**: fdr

False_Negative_Rate
-------------------

False negative rate  is defined as follows:

.. math::

   \text{False Negative Rate} = \frac{False \ Negatives}{False \ Negatives + True \ Positives}

**Source**: `click here <https://fairlearn.org/v0.4.6/api_reference/fairlearn.metrics.html#fairlearn.metrics.false_negative_rate>`__

**Other known names**: miss_rate, fnr, false_non_match_rate

False_Omission_Rate
-------------------

The false omission rate is intuitively the rate at which the classifier will be wrong when labeling an example as negative.

The best value is 0 and the worst value is 1.

.. math::

   \text{False Omission Rate} = \frac{False \ Negatives}{False \ Negatives + True \ Negatives}

**Source**: `click here <https://github.com/credo-ai/credoai_lens/blob/develop/credoai/metrics/metrics_credoai.py>`__

False_Positive_Rate
-------------------

False positive rate is defined as follows:

.. math::

   \text{False Positive Rate} = \frac{False \ Positives}{False \ Positives + True \ Negatives}

**Source**: `click here <https://fairlearn.org/v0.4.6/api_reference/fairlearn.metrics.html#fairlearn.metrics.false_positive_rate>`__

**Other known names**: fpr, false_match_rate, fallout_rate

Gain_Chart
----------



Gini_Coefficient
----------------

Returns the Gini Coefficient of a discriminatory model

    NOTE: There are two popular, yet distinct metrics known as the 'gini coefficient'.

    The value calculated by this function provides a summary statistic for the Cumulative Accuracy Profile (CAP) curve.
    This notion of Gini coefficient (or Gini index) is a _discriminatory_ metric. It helps characterize the ordinal
    relationship between predictions made by a model and the ground truth values for each sample.

    This metric has a linear relationship with the area under the receiver operating characteristic curve:
        :math:`G = 2*AUC - 1`

    See https://towardsdatascience.com/using-the-gini-coefficient-to-evaluate-the-performance-of-credit-score-models-59fe13ef420
    for more details.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_prob : array-like
        Predicted probabilities returned by a call to the model's `predict_proba()` function.

    Returns
    -------
    float
        Discriminatory Gini Coefficient
    

**Other known names**: discriminatory_gini, discriminatory_gini_index, gini_index

Ks_Score_Binary
---------------

Performs the two-sample Kolmogorov-Smirnov test (two-sided) for binary classifiers.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted probabilities returned by the classifier.

    Returns
    -------
    float
        KS statistic value
    

**Other known names**: ks_score

Matthews_Correlation_Coefficient
--------------------------------

The Matthews correlation coefficient is a measure of the quality of a classification model. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html>`__

Max_Error
---------

Max error the maximum residual error, a metric that captures the worst case error between the predicted value and the true value.

In a perfectly fitted single output regression model, ``max_error`` would be 0 on the training set and though this would be highly unlikely in the real world, this metric shows the extent of error that the model had when it was fitted.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html>`__

Max_Proxy_Mutual_Information
----------------------------



Mean_Absolute_Error
-------------------

Mean absolute error is the expected value of the absolute error loss or l1-norm loss.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html>`__

**Other known names**: MAE

Mean_Absolute_Percentage_Error
------------------------------

Mean absolute percentage error is an evaluation metric for regression problems.

The idea of this metric is to be sensitive to relative errors. It is for example not changed by a global scaling of the target variable.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html>`__

Mean_Gamma_Deviance
-------------------

Mean Gamma deviance is the mean `Tweedie deviance <https://en.wikipedia.org/wiki/Tweedie_distribution#The_Tweedie_deviance>`__ error with a power parameter 2. This is a metric that elicits predicted expectation values of regression targets.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_gamma_deviance.html>`__

Mean_Pinball_Loss
-----------------

Mean pinball loss is used to evaluate the predictive performance of quantile regression models. The pinball loss is equivalent to mean_absolute_error when the quantile parameter alpha is set to 0.5.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_pinball_loss.html>`__

Mean_Poisson_Deviance
---------------------

Mean Poisson deviance is the mean `Tweedie deviance <https://en.wikipedia.org/wiki/Tweedie_distribution#The_Tweedie_deviance>`__ error with a power parameter 1. This is a metric that elicits predicted expectation values of regression targets.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_poisson_deviance.html>`__

Mean_Squared_Error
------------------

Mean square error is the expected value of the squared (quadratic) error or loss.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html>`__

**Other known names**: MSD, mean_squared_deviation, MSE

Mean_Squared_Log_Error
----------------------

Mean squared log error is the expected value of the squared logarithmic (quadratic) error or loss.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html>`__

Median_Absolute_Error
---------------------

Median absolute error the median of all absolute differences between the target and the prediction. It is robust to outliers.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html>`__

Membership_Inference_Attack_Score
---------------------------------



Model_Based_Attack_Score
------------------------



Overprediction
--------------

This is the mean of the error where any negative errors (i.e., underpredictions) are set to zero.

**Source**: `click here <https://github.com/fairlearn/fairlearn/blob/main/fairlearn/metrics/_mean_predictions.py>`__

Precision_Recall_Curve
----------------------



**Other known names**: pr_curve

Precision_Score
---------------

Precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

.. math::

   \text{Precision} = \frac{True \ Positives}{True \ Positives + False \ Positives}

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html>`__

**Other known names**: precision

R2_Score
--------

:math:`R^2` (coefficient of determination) regression score function.

Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a :math:`R^2` score of 0.0.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html>`__

**Other known names**: r2, r_squared

Roc_Auc_Score
-------------

ROC-AUC score is the area Under the Receiver Operating Characteristic Curve from prediction scores.

ROC-AUC varies between 0 and 1 (ideal) — with an uninformative classifier yielding 0.5.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html>`__

Roc_Curve
---------



Root_Mean_Squared_Error
-----------------------

Root mean square error is the root of ``mean_squared_error`` metric.

**Source**: `click here <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html>`__

**Other known names**: RMSE

Rule_Based_Attack_Score
-----------------------



Selection_Rate
--------------

Selection rate is the fraction of predicted labels matching the "good" outcome.

**Source**: `click here <https://fairlearn.org/v0.5.0/api_reference/fairlearn.metrics.html#fairlearn.metrics.selection_rate>`__

Sensitive_Feature_Prediction_Score
----------------------------------

Sensitive feature prediction score quantifies how much a model redundantly encoded a sensitive feature.

To evaluate this, a model is trained that tries to predict the sensitive feature from the dataset.

The score ranges from 0.5 - 1.0. If the score is 0.5, the model is random, and no information about the sensitive feature is likely contained in the dataset. A value of 1 means the sensitive feature is able to be perfectly reconstructed.

**Source**: `click here <https://github.com/credo-ai/credoai_lens/blob/develop/credoai/modules/dataset_modules/dataset_fairness.py>`__

Target_Ks_Statistic
-------------------

The two-sample Kolmogorov-Smirnov test (two-sided) statistic for target and prediction arrays
    
The test compares the underlying continuous distributions F(x) and G(x) of two independent samples.
The null hypothesis is that the two distributions are identical, F(x)=G(x)
If the KS statistic is small or the p-value is high, then we cannot reject the null hypothesis in favor of the alternative.

For practical purposes, if the statistic value is higher than `the critical value <https://sparky.rice.edu//astr360/kstest.pdf>`__, the two distributions are different.

**Source**: `click here <https://github.com/credo-ai/credoai_lens/blob/develop/credoai/modules/metrics_credoai.py>`__

**Other known names**: ks_score_regression, ks_score

True_Negative_Rate
------------------

True negative rate (also called specificity or selectivity) refers to the probability of a negative test, conditioned on truly being negative.

.. math::

   \text{True Negative Rate} = \frac{True \ Negatives}{True \ Negatives + False \ Positives }

**Source**: `click here <https://fairlearn.org/v0.5.0/api_reference/fairlearn.metrics.html#fairlearn.metrics.true_negative_rate>`__

**Other known names**: specificity, tnr

True_Positive_Rate
------------------

True Positive Rate (also called sensitivity, recall, or hit rate) refers to the probability of a positive test, conditioned on truly being positive.

**Source**: `click here <https://fairlearn.org/v0.5.0/api_reference/fairlearn.metrics.html#fairlearn.metrics.true_positive_rate>`__

**Other known names**: recall_score, sensitivity, recall, tpr, hit_rate

Underprediction
---------------

This is the mean of the error where any positive errors (i.e. overpredictions) are set to zero.

The absolute value of the underpredictions is used, so the returned value is always positive.

**Source**: `click here <https://github.com/fairlearn/fairlearn/blob/main/fairlearn/metrics/_mean_predictions.py>`__