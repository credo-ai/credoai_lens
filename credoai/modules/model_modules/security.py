import copy
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from art.attacks.extraction import CopycatCNN
from art.estimators.classification import KerasClassifier
from art.estimators.classification.scikitlearn import SklearnClassifier
from credoai.modules.credo_module import CredoModule
from credoai.utils.common import NotRunError
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn import metrics as sk_metrics

tf.compat.v1.disable_eager_execution()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class SecurityModule(CredoModule):
    """Security module for Credo AI.

    This module takes in binary classification model and data and 
     provides functionality to perform security assessment

    Parameters
    ----------
    model : model
        A trained ML model
    x_test : pandas.DataFrame
        The test features
    y_test : pandas.Series
        The test outcome labels
    """

    def __init__(
        self, model, x_test, y_test
    ):
        self.x_test = x_test.to_numpy()
        self.y_test = to_categorical(y_test, num_classes=2)
        self.model = model.model
        self.victim_model = SklearnClassifier(self.model)
        np.random.seed(10)

    def run(self):
        """Runs the assessment process

        Returns
        -------
        dict
            Key: metric name
            Value: metric value
        """
        extraction_performance = self._extraction_attack()

        attack_scores = {
            **extraction_performance
        }

        self.results = attack_scores

        return self

    def prepare_results(self):
        """Prepares results for export to Credo AI's Governance App

        Structures a subset of results for export as a dataframe with appropriate structure
        for exporting. See credoai.modules.credo_module.

        Returns
        -------
        pd.Series

        Raises
        ------
        NotRunError
            If results have not been run, raise
        """
        if self.results is not None:
            return pd.Series(self.results, name='value')
        else:
            raise NotRunError("Results not created yet. Call 'run' to create results")

    def _extraction_attack(self):
        """Model extraction security attack

        In model extraction, the adversary only has access to the prediction API of a target model
            which she queries to extract information about the model internals and train a substitute model.

        Returns
        -------
        dict
            Key: extraction_attack_score
            Value: accuracy of the thieved model / accuracy of the victim model, corrected for chance
        """
        # use half of the test data for model extraction and half for evaluation
        len_steal = int(len(self.x_test)/2)
        indices = np.random.permutation(len(self.x_test))
        x_steal = self.x_test[indices[:len_steal]]
        y_steal = self.y_test[indices[:len_steal]]
        x_test = self.x_test[indices[len_steal:]]
        y_test = self.y_test[indices[len_steal:]]

        # extract
        copycat = CopycatCNN(
            classifier=self.victim_model,
            nb_epochs=5,
            nb_stolen=len_steal
            )

        thieved_model = self._get_model(x_steal.shape[1])
        thieved_classifier = KerasClassifier(thieved_model)

        thieved_classifier = copycat.extract(x_steal, thieved_classifier=thieved_classifier)

        # evaluate 
        y_true = [np.argmax(y, axis=None, out=None) for y in y_test]

        y_pred = [np.argmax(y, axis=None, out=None) for y in thieved_classifier._model.predict(x_test)]
        thieved_classifier_acc = sk_metrics.accuracy_score(y_true, y_pred)

        y_pred = self.victim_model._model.predict(x_test)
        victim_classifier_acc = sk_metrics.accuracy_score(y_true, y_pred)

        metrics = {
            'extraction_attack_score': max((thieved_classifier_acc - 0.5) / (victim_classifier_acc - 0.5), 0)
            }
        
        return metrics

    def _get_model(self, input_dim):
        """Creates a sequential binary classification model 

        Parameters
        ----------
        input_dim : int
            dimension of the feature vector
        """
        model = Sequential()
        model.add(Dense(units=max(int(input_dim/2), 2), input_dim=input_dim, activation='relu'))
        model.add(Dense(units=max(int(input_dim/4), 2), activation='relu'))
        model.add(Dense(2, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return model
