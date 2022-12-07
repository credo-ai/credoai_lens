
Security
========


Security module for Credo AI.

This module takes in classification model and data and provides functionality
to perform security assessment.

The evaluator tests security of the model, by performing 2 types of attacks
(click on the links for more details):

1. `Evasion Attack`_: attempts to create a set of samples that will be
   misclassified by the model
2. `Extraction Attack`_: attempts to infer enough information from the model
   prediction to train a substitutive model.

Parameters
----------
model : model
    A trained binary or multi-class classification model
    The only requirement for the model is to have a `predict` function that returns
    predicted classes for a given feature vectors as a one-dimensional array.
x_train : pandas.DataFrame
    The training features
y_train : pandas.Series
    The training outcome labels
x_test : pandas.DataFrame
    The test features
y_test : pandas.Series
    The test outcome labels

.. _Evasion Attack: https://adversarial-robustness-toolbox.readthedocs.
   io/en/latest/modules/attacks/evasion.html#hopskipjump-attack
.. _Extraction Attack: https://adversarial-robustness-toolbox.readthedocs.
   io/en/latest/modules/attacks/extraction.html#copycat-cnn
