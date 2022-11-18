
Security
========


Security module for Credo AI.

This module takes in classification model and data and
 provides functionality to perform security assessment

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
