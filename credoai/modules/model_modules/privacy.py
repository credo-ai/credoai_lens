from warnings import filterwarnings

import numpy as np
from art.attacks.inference.membership_inference import (
    MembershipInferenceBlackBox,
    MembershipInferenceBlackBoxRuleBased,
)
from art.estimators.classification import BlackBoxClassifier
from credoai.modules.credo_module import CredoModule
from credoai.utils.common import NotRunError
from pandas import Series
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

filterwarnings("ignore")


class PrivacyModule(CredoModule):
    """Privacy module for Credo AI.

    This module takes in in classification model and data and provides functionality
        to perform privacy assessment

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
    """

    def __init__(
        self, model, x_train, y_train, x_test, y_test, attack_train_ratio=0.50
    ):

        self.x_train = x_train.to_numpy()
        self.y_train = y_train.to_numpy()
        self.x_test = x_test.to_numpy()
        self.y_test = y_test.to_numpy()
        self.model = model.model
        self.attack_train_ratio = attack_train_ratio
        self.nb_classes = len(np.unique(self.y_train))
        self.attack_model = BlackBoxClassifier(
            predict_fn=self._predict_binary_class_matrix,
            input_shape=self.x_train[0].shape,
            nb_classes=self.nb_classes,
        )

        self.SUPPORTED_PRIVACY_ATTACKS = {
            "MembershipInferenceBlackBox": {
                "attack": MembershipInferenceBlackBox,
                "type": "model_based",
            },
            "MembershipInferenceBlackBoxRuleBased": {
                "attack": MembershipInferenceBlackBoxRuleBased,
                "type": "rule_based",
            },
        }

        np.random.seed(10)

    def run(self):
        """Runs the assessment process

        Returns
        -------
        dict
            Key: metric name
            Value: metric value
        """

        attack_scores = {}
        for attack_name, attack_info in self.SUPPORTED_PRIVACY_ATTACKS.items():
            attack_scores[attack_name] = self._general_attack_method(attack_info)

        # Best model = worst case
        membership_inference_worst_case = max(attack_scores.values())

        attack_scores[
            "membership_inference_attack_score"
        ] = membership_inference_worst_case

        self.results = attack_scores

        return self

    def prepare_results(self):
        """Prepares results for export to Credo AI's Governance App

        Structures a subset of results for export as a dataframe with appropriate structure
        for exporting. See credoai.modules.credo_module.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        NotRunError
            If results have not been run, raise
        """
        if self.results is not None:
            return Series(self.results, name="value")
        else:
            raise NotRunError("Results not created yet. Call 'run' to create results")

    def _general_attack_method(self, attack_details):
        """
        General wrapper for privacy modules from ART.

        There are 2 types of modules: the ones leveraging machine learning and
        the rule based ones. The former require an extra fit step, and a further
        split of tranining and test so that there is no leakage during the assessment
        phase.

        Parameters
        ----------
        attack_details : dict
            Map of all the supported ART privacy modules and their relative type.

        Returns
        -------
        float
            Accuracy assessment of the attack.
        """
        # Call the main function associated to the attack
        attack = attack_details["attack"](self.attack_model)

        if attack_details["type"] == "rule_based":
            x_train_assess, y_train_assess, x_test_assess, y_test_assess = (
                self.x_train,
                self.y_train,
                self.x_test,
                self.y_test,
            )

        if attack_details["type"] == "model_based":
            train_n = len(self.x_train)
            test_n = len(self.x_test)
            attack_train_size = int(train_n * self.attack_train_ratio)
            attack_test_size = int(test_n * self.attack_train_ratio)
            # generate indices for train/test for attacker
            (
                x_train_attack,
                x_train_assess,
                x_test_attack,
                x_test_assess,
                y_train_attack,
                y_train_assess,
                y_test_attack,
                y_test_assess,
            ) = train_test_split(
                self.x_train,
                self.x_test,
                self.y_train,
                self.y_test,
                train_size=self.attack_train_ratio,
                random_state=42,
            )

            # Split train and test further and fit the model
            attack.fit(x_train_attack, y_train_attack, x_test_attack, y_test_attack)

        # Sets balancing -> This might become optional if we use other metrics, tbd
        x_train_bln, y_train_bln, x_test_bln, y_test_bln = self._balance_sets(
            x_train_assess, y_train_assess, x_test_assess, y_test_assess
        )

        # Attack inference
        train = attack.infer(x_train_bln, y_train_bln)
        test = attack.infer(x_test_bln, y_test_bln)

        return self._assess_attack(train, test, accuracy_score)

    def _predict_binary_class_matrix(self, x):
        """`predict` that returns a binary class matrix

        ----------
        x : features array
            shape (nb_inputs, nb_features)

        Returns
        -------
        numpy.array
            shape (nb_inputs, nb_classes)
        """
        y = self.model.predict(x)
        y_transformed = np.zeros((len(y), self.nb_classes))
        for ai, bi in zip(y_transformed, y):
            ai[bi] = 1
        return y_transformed

    @staticmethod
    def _balance_sets(x_train, y_train, x_test, y_test) -> tuple:
        """
        Balances x and y across train and test sets.

        This is used after any fitting is done, it's needed if we maintain
        the performance score as accuracy. Balancing is done by downsampling the
        greater between train and test.
        """
        if len(x_train) > len(x_test):
            idx = np.random.permutation(len(x_train))[: len(x_test)]
            x_train = x_train[idx]
            y_train = y_train[idx]
        else:
            idx = np.random.permutation(len(x_test))[: len(x_train)]
            x_test = x_test[idx]
            y_test = y_test[idx]
        return x_train, y_train, x_test, y_test

    @staticmethod
    def _assess_attack(train, test, metric) -> float:
        """
        Assess attack using a specific metric.
        """
        y_pred = np.concatenate([train.flatten(), test.flatten()])
        y_true = np.concatenate(
            [
                np.ones(len(train.flatten()), dtype=int),
                np.zeros(len(test.flatten()), dtype=int),
            ]
        )

        return accuracy_score(y_true, y_pred)
