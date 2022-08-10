from warnings import filterwarnings

import numpy as np
from art.attacks.inference.membership_inference import (
    MembershipInferenceBlackBox,
    MembershipInferenceBlackBoxRuleBased,
)
from art.estimators.classification.scikitlearn import SklearnClassifier
from credoai.modules.credo_module import CredoModule
from credoai.utils.common import NotRunError
from pandas import Series
from sklearn.metrics import accuracy_score

filterwarnings("ignore")


class PrivacyModule(CredoModule):
    """Privacy module for Credo AI.

    This module takes in model and data and provides functionality to perform privacy assessment

    Parameters
    ----------
    model : CredoModel
        A trained ML model
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
        self.attack_model = SklearnClassifier(self.model)

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

            attack_train_size = int(len(self.x_train) * self.attack_train_ratio)
            attack_test_size = int(len(self.x_test) * self.attack_train_ratio)

            # Split train and test further and fit the model
            attack.fit(
                self.x_train[:attack_train_size],
                self.y_train[:attack_train_size],
                self.x_test[:attack_test_size],
                self.y_test[:attack_test_size],
            )

            x_train_assess, y_train_assess = (
                self.x_train[attack_train_size:],
                self.y_train[attack_train_size:],
            )
            x_test_assess, y_test_assess = (
                self.x_test[attack_test_size:],
                self.y_test[attack_test_size:],
            )

        # Sets balancing -> This might become optional if we use other metrics, tbd
        x_train_bln, y_train_bln, x_test_bln, y_test_bln = self._balance_sets(
            x_train_assess, y_train_assess, x_test_assess, y_test_assess
        )

        # Attack inference
        train = attack.infer(x_train_bln, y_train_bln)
        test = attack.infer(x_test_bln, y_test_bln)

        return self._assess_attack(train, test, accuracy_score)

    @staticmethod
    def _balance_sets(x_train, y_train, x_test, y_test) -> tuple:
        """
        Balances x and y across train and test sets.

        This is used after any fitting is done, it's needed if we maintain
        the performance score as accuracy. Balancing is done by downsampling the
        greater between train and test.
        """
        if len(x_train) > len(x_test):
            idx = np.random.choice(np.arange(len(x_train)), len(x_test), replace=False)
            x_train = x_train[idx].copy()
            y_train = y_train[idx].copy()
        else:
            idx = np.random.choice(np.arange(len(x_test)), len(x_train), replace=False)
            x_test = x_test[idx].copy()
            y_test = y_test[idx].copy()
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

        return metric(y_true, y_pred)
