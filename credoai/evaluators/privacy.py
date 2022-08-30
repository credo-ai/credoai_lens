from warnings import filterwarnings

import numpy as np
from art.attacks.inference.membership_inference import (
    MembershipInferenceBlackBox,
    MembershipInferenceBlackBoxRuleBased,
)
from art.attacks.inference.attribute_inference import (
    AttributeInferenceBaseline,
    AttributeInferenceBlackBox,
)
from art.estimators.classification import BlackBoxClassifier
from credoai.artifacts import TabularData, ClassificationModel
from credoai.evaluators import Evaluator
from pandas import Series
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from credoai.utils.common import ValidationError

filterwarnings("ignore")

SUPPORTED_MEMBERSHIP_ATTACKS = {
    "MembershipInferenceBlackBoxRuleBased": {
        "attack": {
            "function": MembershipInferenceBlackBoxRuleBased,
            "kwargs": ["classifier"],
        },
        "data_handling": "assess",
        "fit": None,
        "assess": "membership",
    },
    "MembershipInferenceBlackBox": {
        "attack": {
            "function": MembershipInferenceBlackBox,
            "kwargs": ["estimator"],
        },
        "data_handling": "attack-assess",
        "fit": "train_test",
        "assess": "membership",
    },
}
SUPPORTED_ATTRIBUTE_ATTACKS = {
    "AttributeInferenceBaseline": {
        "attack": {
            "function": AttributeInferenceBaseline,
            "kwargs": ["attack_feature"],
        },
        "data_handling": "assess",
        "fit": "train_only",
        "assess": "attribute",
    },
    "AttributeInferenceBlackBox": {
        "attack": {
            "function": AttributeInferenceBlackBox,
            "kwargs": ["estimator", "attack_feature"],
        },
        "data_handling": "assess",
        "fit": "train_only",
        "assess": "attribute",
    },
}


class PrivacyModule(Evaluator):
    """Privacy module for Credo AI.

    This module takes in in classification model and data and provides functionality
        to perform privacy assessment
    """

    def __init__(
        self,
        attack_feature=None,
        attack_feature_name=None,
        attack_train_ratio=0.50,
    ):
        self.attack_train_ratio = attack_train_ratio
        # Validates and assigns attack feature/s
        self._validate_attack_feature(attack_feature, attack_feature_name)

    name = "privacy"

    def __call__(self, model, assessment, training):
        self.model = model
        self.test = assessment
        self.train = training
        # Run validation
        self._validate_arguments()
        # Data prep
        self.x_train = self.train.X.to_numpy()
        self.y_train = self.train.y.to_numpy()
        self.x_test = self.test.X.to_numpy()
        self.y_test = self.test.y.to_numpy()
        self.nb_classes = len(np.unique(self.y_train))
        self.attacked_model = BlackBoxClassifier(
            predict_fn=self._predict_binary_class_matrix,
            input_shape=self.x_train[0].shape,
            nb_classes=self.nb_classes,
        )

        return self

    def evaluate(self):
        """Runs the assessment process

        Returns
        -------
        dict
            Key: metric name
            Value: metric value
        """
        ## TODO: Decide on re-evaluation
        # if self.results:
        #     raise ValueError(
        #         "Evaluation was already run, change override flag to overried"
        #     )

        attacks_to_run = SUPPORTED_MEMBERSHIP_ATTACKS
        if self.attack_feature:
            attacks_to_run = attacks_to_run | SUPPORTED_ATTRIBUTE_ATTACKS

        attack_scores = {}
        for attack_name, attack_info in attacks_to_run.items():
            attack_scores[attack_name] = self._general_attack_method(attack_info)

        # Best model = worst case
        attack_scores["membership_inference_attack_score"] = max(
            [v for k, v in attack_scores.items() if "Membership" in k]
        )

        if self.attack_feature:
            attack_scores["attribute_inference_attack_score"] = max(
                [v for k, v in attack_scores.items() if "Attribute" in k]
            )

        self.results = attack_scores

        return self

    def _validate_arguments(self):
        # Check types, all three are needed -> None is not allowed in this case
        if not isinstance(self.train, TabularData):
            raise ValidationError("""Training data is not of type TabularData.""")
        if not isinstance(self.test, TabularData):
            raise ValidationError("""Test data is not of type TabularData""")
        if not isinstance(self.model, ClassificationModel):
            raise ValidationError("""Model is not of type ClassificationModel.""")
        # Check attack feature in dataset
        if self.attack_feature:
            if not self.attack_feature in self.train.X.columns:
                raise ValidationError(
                    f"Feature {self.attack_feature} not in training data."
                )
            if not self.attack_feature in self.test.X.columns:
                raise ValidationError(
                    f"Feature {self.attack_feature} not in test data."
                )

    def _prepare_results(self):
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
            raise ValueError("Results not created yet. Call 'run' to create results")

    def _general_attack_method(self, attack_details):
        """
        General wrapper for privacy modules from ART.

        There are 2 types of modules: the ones leveraging machine learning and
        the rule based ones. The former require an extra fit step.

        Parameters
        ----------
        attack_details : dict
            Dictionary containing all the attack details

        Returns
        -------
        float
            Accuracy assessment of the attack.
        """
        # Call the main function associated to the attack and pass necessary arguments
        attack = attack_details["attack"]["function"](
            **self._define_model_arguments(attack_details)
        )

        ## Data preparation
        if attack_details["data_handling"] == "assess":
            (
                x_train_assess,
                y_train_assess,
                x_test_assess,
                y_test_assess,
            ) = (self.x_train, self.y_train, self.x_test, self.y_test)
        else:
            attack_assess = self._preprocess_data(
                self.x_train, self.y_train, self.x_test, self.y_test
            )
            (
                x_train_attack,
                x_train_assess,
                y_train_attack,
                y_train_assess,
            ) = attack_assess[0]
            (
                x_test_attack,
                x_test_assess,
                y_test_attack,
                y_test_assess,
            ) = attack_assess[1]

        ## Fit of attack model
        if attack_details["fit"] == "train_test":
            # Split train and test further and fit the model
            attack.fit(x_train_attack, y_train_attack, x_test_attack, y_test_attack)

        if attack_details["fit"] == "train_only":
            attack.fit(x_train_assess)

        ## Rebalancing of the assessment datasets
        x_train_bln, y_train_bln, x_test_bln, y_test_bln = self._balance_sets(
            x_train_assess, y_train_assess, x_test_assess, y_test_assess
        )

        ## Assessment
        if attack_details["assess"] == "membership":
            return self._assess_attack_membership(
                attack, x_train_bln, y_train_bln, x_test_bln, y_test_bln
            )

        if attack_details["assess"] == "attribute":
            return self._assess_attack_attribute(attack, attack_details, x_test_bln)

    def _define_model_arguments(self, attack_details):
        """
        Collates the arguments to feed to the attack initialization.

        Parameters
        ----------
        attack_details : dict
            Dictionary containing all the attack details

        Returns
        -------
        dict
            Named arguments dictionary for the attack function
        """
        arg_dict = {
            "estimator": self.attacked_model,
            "classifier": self.attacked_model,
            "attack_feature": self.attack_feature,
        }
        return {i: arg_dict[i] for i in attack_details["attack"]["kwargs"]}

    def _preprocess_data(self, *args) -> tuple:
        """
        Further split test and train dataset.

        Parameters
        ----------
        args : dict
            x_train, y_train, x_test, y_test. The order needs to be respected.

        Returns
        -------
        tuple
            Length 2 tuple, first elements contains the split of the train set,
            the second element contains the split of the test set.
        """

        train_sets = train_test_split(
            args[0], args[1], random_state=42, train_size=self.attack_train_ratio
        )
        test_sets = train_test_split(
            args[2], args[3], random_state=42, train_size=self.attack_train_ratio
        )
        return (train_sets, test_sets)

    def _assess_attack_attribute(self, attack, attack_details, x_test_bln) -> float:
        """
        Assess attack result for attribute type attack.

        A comparison between the original feature and the inferred one.

        Parameters
        ----------
        attack :
            ART attack model ready for inference
        attack_details : dict
            Dictionary containing all the attack details
        x_test_bln : numpy.array
            Balanced test dataset

        Returns
        -------
        float
            Accuracy of the attack
        """
        # Compare infered feature with original
        extra_arg = {}
        if "estimator" in attack_details["attack"]["kwargs"]:
            original_model_pred = np.array(
                [np.argmax(arr) for arr in self.model.predict(x_test_bln)]
            ).reshape(-1, 1)
            # Pass this to model inference
            extra_arg = {"pred": original_model_pred}

        # Compare original feature with the one deduced by the model
        original = x_test_bln[:, self.attack_feature].copy()
        inferred = attack.infer(
            np.delete(x_test_bln, self.attack_feature, 1), **extra_arg
        )
        return np.sum(inferred == original) / len(inferred)

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

    def _validate_attack_feature(self, attack_feature, attack_feature_name):
        if isinstance(attack_feature, int) and attack_feature_name is None:
            raise ValidationError("attack_feature_name must be provided")

        self.attack_feature_name = attack_feature_name
        self.attack_feature = attack_feature

    @staticmethod
    def _assess_attack_membership(
        attack, x_train_bln, y_train_bln, x_test_bln, y_test_bln
    ) -> float:
        """
        Assess attack using a specific metric.
        """
        train = attack.infer(x_train_bln, y_train_bln)
        test = attack.infer(x_test_bln, y_test_bln)
        y_pred = np.concatenate([train.flatten(), test.flatten()])
        y_true = np.concatenate(
            [
                np.ones(len(train.flatten()), dtype=int),
                np.zeros(len(test.flatten()), dtype=int),
            ]
        )
        return accuracy_score(y_true, y_pred)

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
