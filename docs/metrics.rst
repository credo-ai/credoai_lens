Metrics
=======

Lens supports many metrics out-of-the-box. These metrics can be accessed simply, 
by providing their name as a string to the Lens's assessment spec. 
Below are some of the metrics we support. For a comprehensive list, 
the following can be run in your python environment:

::

   from credoai.metrics import list_metrics
   list_metrics()


Other metrics are easily incorporated by using the `Metric` class, which can accommodate 
any assessment function.

Metric list
-----------

``accuracy_score``

Accuracy is the fraction of predictions that a classification model got right. This metric is not robust to class imbalance.

The best value is 1 and the worst value is 0.

.. math::

   \text{Accuracy} = \frac{Correct \ Predictions}{Total \ Number \ of \ Examples}

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html>`__)

------------

``average_precision_score``

Average precision summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html>`__)

------------

``balanced_accuracy_score``

The balanced accuracy in classification problems is used to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.

The best value is 1 and the worst value is 0.

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html>`__)

------------

``d2_tweedie_score``

:math:`D^2`  regression score is percentage of `Tweedie deviance <https://en.wikipedia.org/wiki/Tweedie_distribution#The_Tweedie_deviance>`__ explained.

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.d2_tweedie_score.html>`__)

------------

``demographic_parity_difference``

Demographic parity difference is a parity metric that is satisfied if the results of a model's classification are not dependent on a given sensitive attribute.

Demographic parity difference should be ideally 0.

Equivalents: ``statistical_parity``, ``demographic_parity``

(`source <https://fairlearn.org/v0.4.6/api_reference/fairlearn.metrics.html#fairlearn.metrics.demographic_parity_difference>`__)

------------

``demographic_parity_ratio``

Demographic parity ratio is a parity metric that is satisfied if the results of a model's classification are not dependent on a given sensitive attribute.

Demographic parity ratio should be ideally 1.

Equivalents: ``disparate_impact``

(`source <https://fairlearn.org/v0.4.6/api_reference/fairlearn.metrics.html#fairlearn.metrics.demographic_parity_ratio>`__)

------------

``equalized_odds_difference``

The equalized odds difference of 0 means that all groups have the same true positive, true negative, false positive, and false negative rates.

Equivalents: ``equalized_odds``

(`source <https://fairlearn.org/v0.4.6/api_reference/fairlearn.metrics.html#fairlearn.metrics.equalized_odds_difference>`__)

------------

``equal_opportunity_difference``

The equalized odds difference is equivalent to the `true_positive_rate_difference` defined as the difference between the largest and smallest of :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a` of the sensitive feature(s).

Equivalents: ``equal_opportunity``

(`source <https://github.com/credo-ai/credoai_lens/blob/develop/credoai/metrics/credoai_metrics.py>`__)

------------

``explained_variance_score``

Explained variance regression score function.

Best possible score is 1.0, lower values are worse.

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html>`__)

------------

``f1_score``

Also known as balanced F-score or F-measure, the F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal.

.. math::

   \text{False Positive Rate} = \frac{2 \times Precision \times Recall}{Precision + Recall}

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>`__)

------------

``false_discovery_rate``

False discovery rate is intuitively the rate at which the classifier will be wrong when labeling an example as positive.

The best value is 0 and the worst value is 1.

.. math::

   \text{False Discovery Rate} = \frac{False \ Positives}{False \ Positives + True \ Positives}

Equivalents: ``fdr``

(`source <https://github.com/credo-ai/credoai_lens/blob/develop/credoai/metrics/credoai_metrics.py>`__)

------------

``false_negative_rate``

False negative rate  is defined as follows:

.. math::

   \text{False Negative Rate} = \frac{False \ Negatives}{False \ Negatives + True \ Positives}

Equivalents: ``fnr``, ``miss_rate``

(`source <https://fairlearn.org/v0.4.6/api_reference/fairlearn.metrics.html#fairlearn.metrics.false_negative_rate>`__)

------------

``false_omission_rate``

The false omission rate is intuitively the rate at which the classifier will be wrong when labeling an example as negative.

The best value is 0 and the worst value is 1.

.. math::

   \text{False Omission Rate} = \frac{False \ Negatives}{False \ Negatives + True \ Negatives}

(`source <https://github.com/credo-ai/credoai_lens/blob/develop/credoai/metrics/credoai_metrics.py>`__)

------------

``false_positive_rate``

False positive rate is defined as follows:

.. math::

   \text{False Positive Rate} = \frac{False \ Positives}{False \ Positives + True \ Negatives}

Equivalents: ``fpr``, ``fallout_rate``

(`source <https://fairlearn.org/v0.4.6/api_reference/fairlearn.metrics.html#fairlearn.metrics.false_positive_rate>`__)

------------

``matthews_correlation_coefficient``

The Matthews correlation coefficient is a measure of the quality of a classification model. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html>`__)

------------

``max_error``

Max error the maximum residual error, a metric that captures the worst case error between the predicted value and the true value.

In a perfectly fitted single output regression model, ``max_error`` would be 0 on the training set and though this would be highly unlikely in the real world, this metric shows the extent of error that the model had when it was fitted.

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html>`__)

------------

``mean_absolute_error``

Mean absolute error is the expected value of the absolute error loss or l1-norm loss.

Equivalents: ``MAE``

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html>`__)

------------

``mean_absolute_percentage_error``

Mean absolute percentage error is an evaluation metric for regression problems.

The idea of this metric is to be sensitive to relative errors. It is for example not changed by a global scaling of the target variable.

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html>`__)

------------

``mean_gamma_deviance``

Mean Gamma deviance is the mean `Tweedie deviance <https://en.wikipedia.org/wiki/Tweedie_distribution#The_Tweedie_deviance>`__ error with a power parameter 2. This is a metric that elicits predicted expectation values of regression targets.

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_gamma_deviance.html>`__)

------------

``mean_pinball_loss``

Mean pinball loss is used to evaluate the predictive performance of quantile regression models. The pinball loss is equivalent to mean_absolute_error when the quantile parameter alpha is set to 0.5.

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_pinball_loss.html>`__)

------------

``mean_poisson_deviance``

Mean Poisson deviance is the mean `Tweedie deviance <https://en.wikipedia.org/wiki/Tweedie_distribution#The_Tweedie_deviance>`__ error with a power parameter 1. This is a metric that elicits predicted expectation values of regression targets.

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_poisson_deviance.html>`__)

------------

``mean_squared_error``

Mean square error is the expected value of the squared (quadratic) error or loss.

Equivalents: ``MSE``, ``MSD``, ``mean_squared_deviation``

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html>`__)

------------

``mean_squared_log_error``

Mean squared log error is the expected value of the squared logarithmic (quadratic) error or loss.

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html>`__)

------------

``median_absolute_error``

Median absolute error the median of all absolute differences between the target and the prediction. It is robust to outliers.

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.median_absolute_error.html>`__)

------------

``overprediction``

This is the mean of the error where any negative errors (i.e., underpredictions) are set to zero.

(`source <https://github.com/fairlearn/fairlearn/blob/main/fairlearn/metrics/_mean_predictions.py>`__)

------------

``precision_score``

Precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

.. math::

   \text{Precision} = \frac{True \ Positives}{True \ Positives + False \ Positives}

Equivalents: ``precision``

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html>`__)

------------

``r2_score``

:math:`R^2` (coefficient of determination) regression score function.

Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a :math:`R^2` score of 0.0.

Equivalents: ``r_squared``, ``r2``

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html>`__)

------------

``roc_auc_score``

ROC-AUC score is the area Under the Receiver Operating Characteristic Curve from prediction scores.

ROC-AUC varies between 0 and 1 (ideal) — with an uninformative classifier yielding 0.5.

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html>`__)

------------

``root_mean_squared_error``

Root mean square error is the root of ``mean_squared_error`` metric.

Equivalents: ``RMSE``

(`source <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html>`__)

------------

``selection_rate``

Selection rate is the fraction of predicted labels matching the "good" outcome.

(`source <https://fairlearn.org/v0.5.0/api_reference/fairlearn.metrics.html#fairlearn.metrics.selection_rate>`__)

------------

``sensitive_feature_prediction_score``

Sensitive feature prediction score quantifies how much a model redundantly encoded a sensitive feature.

To evaluate this, a model is trained that tries to predict the sensitive feature from the dataset.

The score ranges from 0.5 - 1.0. If the score is 0.5, the model is random, and no information about the sensitive feature is likely contained in the dataset. A value of 1 means the sensitive feature is able to be perfectly reconstructed.

(`source <https://github.com/credo-ai/credoai_lens/blob/develop/credoai/modules/dataset_modules/dataset_fairness.py>`__)

------------

``target_ks_statistic``

The two-sample Kolmogorov-Smirnov test (two-sided) statistic for target and prediction arrays
    
The test compares the underlying continuous distributions F(x) and G(x) of two independent samples.
The null hypothesis is that the two distributions are identical, F(x)=G(x)
If the KS statistic is small or the p-value is high, then we cannot reject the null hypothesis in favor of the alternative.

For practical purposes, if the statistic value is higher than `the critical value <https://sparky.rice.edu//astr360/kstest.pdf>`__, the two distributions are different.

(`source <https://github.com/credo-ai/credoai_lens/blob/develop/credoai/metrics/credoai_metrics.py>`__)

------------

``true_negative_rate``

True negative rate (also called specificity or selectivity) refers to the probability of a negative test, conditioned on truly being negative.

.. math::

   \text{True Negative Rate} = \frac{True \ Negatives}{True \ Negatives + False \ Positives }

Equivalents: ``tnr``, ``specificity``

(`source <https://fairlearn.org/v0.5.0/api_reference/fairlearn.metrics.html#fairlearn.metrics.true_negative_rate>`__)

------------

``true_positive_rate``

True Positive Rate (also called sensitivity, recall, or hit rate) refers to the probability of a positive test, conditioned on truly being positive.

Equivalents: ``tpr``, ``recall_score``, ``recall``, ``sensitivity``, ``hit_rate``

(`source <https://fairlearn.org/v0.5.0/api_reference/fairlearn.metrics.html#fairlearn.metrics.true_positive_rate>`__)

------------

``underprediction``

This is the mean of the error where any positive errors (i.e. overpredictions) are set to zero.

The absolute value of the underpredictions is used, so the returned value is always positive.

(`source <https://github.com/fairlearn/fairlearn/blob/main/fairlearn/metrics/_mean_predictions.py>`__)
