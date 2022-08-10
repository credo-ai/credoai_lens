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
from sklearn import metrics as sk_metrics

filterwarnings("ignore")


class PrivacyModule(CredoModule):
    """Privacy module for Credo AI.

    This module takes in model and data and provides functionality to perform privacy assessment

    Parameters
    ----------
    # TODO: Double check with Ian if types are correct for usage in Lens
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
        np.random.seed(10)

    @staticmethod
    def balance_sets(x_train, y_train, x_test, y_test) -> tuple:
        """
        Balances x and y across train and test sets.

        This is used after any fitting is done, it's needed if we maintain
        the performance score as accuracy. Balancing is done by downsampling the
        greater between train and test.
        """
        if len(x_train) > len(x_test):
            idx = np.random.choice(np.arange(len(x_train)), len(x_test), replace=False)
            x_train = x_train[idx]
            y_train = y_train[idx]
        else:
            idx = np.random.choice(np.arange(len(x_test)), len(x_train), replace=False)
            x_test = x_test[idx]
            y_test = y_test[idx]
        return x_train, y_train, x_test, y_test

    @staticmethod
    def assess_attack(train, test, metric) -> float:
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

    def run(self):
        """Runs the assessment process

        Returns
        -------
        dict
            Key: metric name
            Value: metric value
        """

        attack_scores = {
            "rule_based_attack_score": self._rule_based_attack(),
            "model_based_attack_score": self._model_based_attack(),
        }
        membership_inference_worst_case = max(
            attack_scores["rule_based_attack_score"],
            attack_scores["model_based_attack_score"],
        )
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

    def _rule_based_attack(self):
        """Rule-based privacy attack

        The rule-based attack uses the simple rule to determine membership in the training data:
            if the model's prediction for a sample is correct, then it is a member.
            Otherwise, it is not a member.

        Returns
        -------
        dict
            Key: rule_based_attack_accuracy_score
            Value: membership prediction accuracy of the rule-based attack
        """
        attack = MembershipInferenceBlackBoxRuleBased(self.attack_model)

        # Sets balancing
        x_train_bln, y_train_bln, x_test_bln, y_test_bln = self.balance_sets(
            self.x_train, self.y_train, self.x_test, self.y_test
        )

        # Attack inference
        train = attack.infer(x_train_bln, y_train_bln)
        test = attack.infer(x_test_bln, y_test_bln)

        return self.assess_attack(train, test, sk_metrics.accuracy_score)

    def _model_based_attack(self):
        """Model-based privacy attack

        The model-based attack trains an additional classifier (called the attack model)
            to predict the membership status of a sample. It can use as input to the learning process
            probabilities/logits or losses, depending on the type of model and provided configuration.

        Returns
        -------
        dict
            Key: model_based_attack_accuracy_score
            Value: membership prediction accuracy of the model-based attack
        """
        attack_train_size = int(len(self.x_train) * self.attack_train_ratio)
        attack_test_size = int(len(self.x_test) * self.attack_train_ratio)

        attack = MembershipInferenceBlackBox(self.attack_model)

        # train attack model
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

        # Sets balancing
        x_train_bln, y_train_bln, x_test_bln, y_test_bln = self.balance_sets(
            x_train_assess, y_train_assess, x_test_assess, y_test_assess
        )

        # Attack inference
        train = attack.infer(x_train_bln, y_train_bln)
        test = attack.infer(x_test_bln, y_test_bln)

        return self.assess_attack(train, test, sk_metrics.accuracy_score)
