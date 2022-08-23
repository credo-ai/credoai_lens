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
from credoai.modules.credo_module import CredoModule
from credoai.utils.common import NotRunError
from pandas import Series
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
    attack_train_ratio : float
        If further split of the data is required to create a validation set,
        this decides the ratio
    attack_feature : int,list
        For attribute inference: the index of the attribute to be inferred
        in the original dataset.
    """

    def __init__(
        self,
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        attack_train_ratio=0.50,
        attack_feature=None,
    ):

        self.x_train = x_train.to_numpy()
        self.y_train = y_train.to_numpy()
        self.x_test = x_test.to_numpy()
        self.y_test = y_test.to_numpy()
        self.model = model
        self.attack_train_ratio = attack_train_ratio
        self.attack_feature = self._validate_attack_feature(attack_feature)
        self.nb_classes = len(np.unique(self.y_train))

        self.attacked_model = BlackBoxClassifier(
            predict_fn=self._predict_binary_class_matrix,
            input_shape=self.x_train[0].shape,
            nb_classes=self.nb_classes,
        )

    def run(self):
        """Runs the assessment process

        Returns
        -------
        dict
            Key: metric name
            Value: metric value
        """
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

    @staticmethod
    def _validate_attack_feature(attack_feature):
        if isinstance(attack_feature, int) or attack_feature is None:
            return attack_feature
        else:
            raise ValueError(
                f"The argument attack_feature is expected to be int. Current type {type(attack_feature)}"
            )

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
