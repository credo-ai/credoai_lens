from typing import Optional, Union
from warnings import filterwarnings

import numpy as np
from art.attacks.inference.attribute_inference import (
    AttributeInferenceBaseline,
    AttributeInferenceBlackBox,
)
from art.attacks.inference.membership_inference import (
    MembershipInferenceBlackBox,
    MembershipInferenceBlackBoxRuleBased,
)
from art.estimators.classification import BlackBoxClassifier
from connect.evidence import MetricContainer
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from credoai.artifacts import ClassificationModel, DummyClassifier, TabularData
from credoai.evaluators.evaluator import Evaluator
from credoai.evaluators.utils.validation import (
    check_data_for_nulls,
    check_data_instance,
    check_feature_presence,
    check_model_instance,
    check_requirements_existence,
)
from credoai.utils.common import ValidationError

filterwarnings("ignore")

SUPPORTED_MEMBERSHIP_ATTACKS = {
    "MembershipInference-BlackBoxRuleBased": {
        "attack": {
            "function": MembershipInferenceBlackBoxRuleBased,
            "kwargs": ["classifier"],
        },
        "data_handling": "assess",
        "fit": None,
        "assess": "membership",
    },
    "MembershipInference-BlackBox": {
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
    "AttributeInference-Baseline": {
        "attack": {
            "function": AttributeInferenceBaseline,
            "kwargs": ["attack_feature"],
        },
        "data_handling": "assess",
        "fit": "train_only",
        "assess": "attribute",
    },
    "AttributeInference-BlackBox": {
        "attack": {
            "function": AttributeInferenceBlackBox,
            "kwargs": ["estimator", "attack_feature"],
        },
        "data_handling": "assess",
        "fit": "train_only",
        "assess": "attribute",
    },
}


class Privacy(Evaluator):
    """
    Privacy module for Credo AI (Experimental)

    This module takes  provides functionality to perform privacy assessment.

    The main library leveraged for the purpose is the
    `adversarial robustness toolbox <https://adversarial-robustness-toolbox.readthedocs.io/en/latest/>`_.
    The types of attacks used by this evaluator are the following (click on the links for more info):

    * `Attribute Inference Baseline`_: Trains a neural network to learn the attacked feature from the other features.
    * `Attribute Inference BlackBox`_: Trains a neural network to learn the attacked feature from the other features and
      the model's prediction.
    * `Membership Inference BlackBox`_: Trains a neural network to assess if some records were used for the model training.
    * `Membership Inference BlackBox Rule Based`_: Use a simple rule based approach to assess if some records
      were used for the model training.

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
    attack_feature : Union[str, int, None], optional
        Either the name or the column number of the feature to be attacked. If the column
        number is provided, the following parameter `attack_feature_name` needs to be provided.
        Default is None, in this case no attack feature is performed.
    attack_feature_name : Optional[str], optional
        The name of the feature to be attacked, this is to be provided only in the case `attack_feature` is
        an integer. This allows for data like numpy.matrix that do not possess column names can be passed
        as datasets. By default None.
    attack_train_ratio : float, optional
        Internally the train/test dataset are further split in order to train the models performing the
        attacks. This indicates the split ratio, by default 0.50

    .. _Attribute Inference Baseline: https://adversarial-robustness-toolbox.readthedocs.
       io/en/latest/modules/attacks/inference/attribute_inference.html#attribute-inference-baseline
    .. _Attribute Inference BlackBox: https://adversarial-robustness-toolbox.readthedocs.
       io/en/latest/modules/attacks/inference/attribute_inference.html#attribute-inference-black-box
    .. _Membership Inference BlackBox Rule Based: https://adversarial-robustness-toolbox.readthedocs.
       io/en/latest/modules/attacks/inference/membership_inference.html#membership-inference-black-box-rule-based
    .. _Membership Inference BlackBox: https://adversarial-robustness-toolbox.readthedocs.
       io/en/latest/modules/attacks/inference/membership_inference.html#membership-inference-black-box
    """

    required_artifacts = {"model", "assessment_data", "training_data"}

    def __init__(
        self,
        attack_feature: Union[str, int, None] = None,
        attack_feature_name: Optional[str] = None,
        attack_train_ratio: float = 0.50,
    ):

        self.attack_train_ratio = attack_train_ratio
        # Validates and assigns attack feature/s
        self._validate_attack_feature(attack_feature, attack_feature_name)
        super().__init__()

    def _validate_arguments(self):
        """
        Input validation step, this is run after _init_artifacts() in the
        parent class.
        """
        check_requirements_existence(self)
        check_model_instance(self.model, (ClassificationModel, DummyClassifier))
        for ds in ["assessment_data", "training_data"]:
            artifact = vars(self)[ds]
            check_data_instance(artifact, TabularData, ds)
            check_data_for_nulls(artifact, ds)
            if isinstance(self.attack_feature, str):
                check_feature_presence(
                    self.attack_feature, artifact.X, "assessment_data"
                )

    def _setup(self):
        """
        Complete initialization after the artifacts have been passed by _init_artifacts() in the
        parent class.
        """
        # Data prep
        self.x_train = self.training_data.X.to_numpy()
        self.y_train = self.training_data.y.to_numpy()
        self.x_test = self.assessment_data.X.to_numpy()
        self.y_test = self.assessment_data.y.to_numpy()
        if isinstance(self.attack_feature, str):
            (
                self.attack_feature_name,
                self.attack_feature,
            ) = self.attack_feature, self.training_data.X.columns.get_loc(
                self.attack_feature
            )
        self.nb_classes = len(np.unique(self.y_train))
        self.attacked_model = BlackBoxClassifier(
            predict_fn=self._predict_binary_class_matrix,
            input_shape=self.x_train[0].shape,
            nb_classes=self.nb_classes,
        )

        return self

    def evaluate(self):
        """
        Runs the assessment process.

        Returns
        -------
            Update the results with a list of MetricContainers

        """
        # Define attacks to run based on init parameters
        attacks_to_run = SUPPORTED_MEMBERSHIP_ATTACKS
        if self.attack_feature is not None:
            attacks_to_run = {**attacks_to_run, **SUPPORTED_ATTRIBUTE_ATTACKS}

        # Run all attacks
        attack_scores = {}
        for attack_name, attack_info in attacks_to_run.items():
            attack_scores[attack_name] = self._general_attack_method(attack_info)

        self.results = self._format_scores(attack_scores)

        return self

    def _format_scores(self, attack_scores: dict):
        """
        Takes all results, defines the best model and returns the container

        Parameters
        ----------
        attack_scores : dict
            Results of the inferences.
        """

        # Select overall scores for each type of attacks
        attack_scores["MembershipInference-attack_score"] = max(
            [v for k, v in attack_scores.items() if "Membership" in k]
        )

        if self.attack_feature is not None:
            attack_scores["AttributeInference-attack_score"] = max(
                [v for k, v in attack_scores.items() if "Attribute" in k]
            )

        attack_score = DataFrame(list(attack_scores.items()), columns=["type", "value"])
        attack_score[["type", "subtype"]] = attack_score.type.str.split(
            "-", expand=True
        )
        attack_score = [MetricContainer(attack_score, **self.get_info())]

        return attack_score

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

        ## Re-balancing of the assessment datasets
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
        # Compare inferred feature with original
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
        """
        `predict` that returns a binary class matrix.

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

    def _validate_attack_feature(
        self, attack_feature: Union[str, int, None], attack_feature_name: Optional[str]
    ):
        """
        Validation of attack feature.

        Parameters
        ----------
        attack_feature : Union[str, int, None]
            Feature name or position in the dataframe
        attack_feature_name : Optional[str]
            Feature name

        Raises
        ------
        ValidationError
            If attack feature is positional a correspondent name needs to be provided.
        """
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
        the performance score as accuracy. Balancing is done by down sampling the
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
