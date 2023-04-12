import os

import numpy as np
import pandas as pd
import tensorflow as tf
from art.attacks.evasion import HopSkipJump
from art.attacks.extraction import CopycatCNN
from art.estimators.classification import BlackBoxClassifier, TensorFlowV2Classifier
from connect.evidence import MetricContainer
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn import metrics as sk_metrics
from sklearn.metrics import pairwise
from sklearn.preprocessing import StandardScaler

from credoai.artifacts.data.tabular_data import TabularData
from credoai.artifacts.model.classification_model import (
    ClassificationModel,
    DummyClassifier,
)
from credoai.evaluators.evaluator import Evaluator
from credoai.evaluators.utils.validation import (
    check_data_for_nulls,
    check_data_instance,
    check_model_instance,
    check_requirements_existence,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Security(Evaluator):
    """
    Security module for Credo AI. (Experimental)

    This module takes in classification model and data and provides functionality
    to perform security assessment.

    The evaluator tests security of the model, by performing 2 types of attacks
    (click on the links for more details):

    1. `Evasion Attack`_: attempts to create a set of samples that will be
       misclassified by the model
    2. `Extraction Attack`_: attempts to infer enough information from the model
       prediction to train a substitutive model.

    Required Artifacts
    ------------------
        **Required Artifacts**

        Generally artifacts are passed directly to :class:`credoai.lens.Lens`, which
        handles evaluator setup. However, if you are using the evaluator directly, you
        will need to pass the following artifacts when instantiating the evaluator:

        - model: :class:`credoai.artifacts.Model`
        - assessment_data: :class:`credoai.artifacts.TabularData`
        - training_data: :class:`credoai.artifacts.TabularData`

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
    """

    required_artifacts = {"model", "assessment_data", "training_data"}

    def _validate_arguments(self):
        check_requirements_existence(self)
        check_model_instance(self.model, (ClassificationModel, DummyClassifier))
        for ds in ["assessment_data", "training_data"]:
            artifact = vars(self)[ds]
            check_data_instance(artifact, TabularData, ds)
            check_data_for_nulls(artifact, ds)

    def _setup(self):
        self.x_train = self.training_data.X.to_numpy()
        self.y_train = self.training_data.y
        self.nb_classes = len(np.unique(self.y_train))
        self.x_test = self.assessment_data.X.to_numpy()
        self.y_test = to_categorical(
            self.assessment_data.y, num_classes=self.nb_classes
        )
        self.victim_model = BlackBoxClassifier(
            predict_fn=self._predict_binary_class_matrix,
            input_shape=self.x_train[0].shape,
            nb_classes=self.nb_classes,
        )
        np.random.seed(10)
        return self

    def evaluate(self):
        """
        Runs the assessment process

        Returns
        -------
        dict
            Key: metric name
            Value: metric value
        """
        # tf.compat.v1.disable_eager_execution()
        res = {**self._extraction_attack(), **self._evasion_attack()}
        res = pd.DataFrame(list(res.items()), columns=["type", "value"])
        res[["type", "subtype"]] = res.type.str.split("-", expand=True)
        self.results = [MetricContainer(res, **self.get_info())]
        # tf.compat.v1.enable_eager_execution()
        return self

    def _extraction_attack(self):
        """
        Model extraction security attack

        In model extraction, the adversary only has access to the prediction API of a target model
        which she queries to extract information about the model internals and train a substitute model.

        Returns
        -------
        dict
            Key: extraction_attack_score
            Value: accuracy of the thieved model / accuracy of the victim model, corrected for chance
        """
        # use half of the test data for model extraction and half for evaluation
        len_steal = int(len(self.x_test) / 2)
        indices = np.random.permutation(len(self.x_test))
        x_steal = self.x_test[indices[:len_steal]]
        y_steal = self.y_test[indices[:len_steal]]
        x_test = self.x_test[indices[len_steal:]]
        y_test = self.y_test[indices[len_steal:]]

        # extract
        copycat = CopycatCNN(
            classifier=self.victim_model, nb_epochs=5, nb_stolen=len_steal
        )

        def my_train_step(model, images, labels):
            return model.train_step((images, labels))

        thieved_model = self._get_model(x_steal.shape[1])
        thieved_classifier = TensorFlowV2Classifier(
            thieved_model,
            nb_classes=self.nb_classes,
            input_shape=x_steal.shape[1],
            loss_object=thieved_model.loss,
            train_step=my_train_step,
        )

        thieved_classifier = copycat.extract(
            x_steal, thieved_classifier=thieved_classifier
        )

        # evaluate
        y_true = [np.argmax(y, axis=None, out=None) for y in y_test]

        y_pred = [
            np.argmax(y, axis=None, out=None)
            for y in thieved_classifier._model.predict(x_test)
        ]
        thieved_classifier_acc = sk_metrics.accuracy_score(y_true, y_pred)

        y_pred = [
            np.argmax(y, axis=None, out=None) for y in self.victim_model.predict(x_test)
        ]
        victim_classifier_acc = sk_metrics.accuracy_score(y_true, y_pred)

        metrics = {
            "extraction-attack_score": max(
                (thieved_classifier_acc - 0.5) / (victim_classifier_acc - 0.5), 0
            )
        }

        return metrics

    def _get_model(self, input_dim):
        """
        Creates a sequential binary classification model

        Parameters
        ----------
        input_dim : int
            dimension of the feature vector
        """
        model = Sequential()
        model.add(
            Dense(
                units=max(int(input_dim / 2), self.nb_classes),
                input_dim=input_dim,
                activation="relu",
            )
        )
        model.add(
            Dense(units=max(int(input_dim / 4), self.nb_classes), activation="relu")
        )
        model.add(Dense(self.nb_classes))
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=["accuracy"],
        )

        return model

    def _evasion_attack(self, nsamples=10, distance_threshold=0.1):
        """
        Model evasion security attack

        In model evasion, the adversary only has access to the prediction API of a target model
            which she queries to create minimally-perturbed samples that get misclassified
            by the model.

        Parameters
        ----------
        nsamples : int
            number of samples to attack
        distance_threshold : float
            Euclidean distance threshold between an adversarial sample and its original sample
             normalized by the sample length. An adversarial sample more distant than
             this is considered a failed attempt.

        Returns
        -------
        dict
            Key: evasion_attack_score
            Value: evasion success rate given a distance threshold
        """
        hsj = HopSkipJump(classifier=self.victim_model)

        origl_sample = self.x_test[0:nsamples]
        adver_sample = hsj.generate(origl_sample)

        origl_pred = [
            np.argmax(y, axis=None, out=None)
            for y in self.victim_model.predict(origl_sample)
        ]
        adver_pred = [
            np.argmax(y, axis=None, out=None)
            for y in self.victim_model.predict(adver_sample)
        ]

        # standardize for robust distance calculation
        scaler = StandardScaler()
        scaler.fit(self.x_train)
        origl_sample_scaled = scaler.transform(origl_sample)
        adver_sample_scaled = scaler.transform(adver_sample)

        metrics = {
            "evasion-attack_score": self._evasion_success_rate(
                origl_pred,
                adver_pred,
                origl_sample_scaled,
                adver_sample_scaled,
                distance_threshold,
            )
        }

        return metrics

    def _evasion_success_rate(
        self,
        origl_pred,
        adver_pred,
        origl_sample_scaled,
        adver_sample_scaled,
        distance_threshold=0.1,
    ):
        """
        Calculates evasion success rate

        Parameters
        ----------
        origl_pred : list
            predictions of the original samples
        adver_pred : list
            predictions of the adversarial samples
        origl_sample_scaled : list
            scaled original samples
        adver_sample_scaled : list
            scaled adversarial samples
        distance_threshold : float
            Euclidean distance threshold between an adversarial sample and its original sample
             normalized by the sample length. An adversarial sample more distant than
             this is considered a failed attempt.

        Returns
        -------
        float
            the proportion of the predictions that have been flipped and
             are not distant
        """
        length = len(origl_sample_scaled)
        distances = (
            np.diag(
                pairwise.euclidean_distances(origl_sample_scaled, adver_sample_scaled)
            )
            / length
        )
        idx = np.where(distances <= distance_threshold)
        origl_pred = np.array(origl_pred)
        adver_pred = np.array(adver_pred)
        if origl_pred[idx].size > 0:
            return (
                np.count_nonzero(np.not_equal(origl_pred[idx], adver_pred[idx]))
                / length
            )
        else:
            return 0

    def _predict_binary_class_matrix(self, x):
        """
        `predict` that returns a binary class matrix

        Parameters
        ----------
        x : features array
            shape (nb_inputs, nb_features)

        Returns
        -------
        numpy.array
            shape (nb_inputs, nb_classes)
        """
        y = self.model.predict(x)
        y_transformed = np.zeros((len(x), self.nb_classes))
        for ai, bi in zip(y_transformed, y):
            ai[bi] = 1
        return y_transformed
