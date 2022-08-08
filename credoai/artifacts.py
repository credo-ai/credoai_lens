"""Artifacts used by Lens to structure governance, models and data"""
# CredoLens relies on four classes
# - CredoGovernance
# - CredoModel
# - CredoData
# - CredoAssessment

# CredoGovernance contains the information needed
# to connect CredoLens with the Credo AI Governance App

# CredoModel follows an `adapter pattern` to convert
# any model into an interface CredoLens can work with.
# The functionality contained in CredoModel determines
# what assessments can be run.

# CredoData is a lightweight wrapper that stores data

# CredoAssessment is the interface between a CredoModel,
# CredoData and a module, which performs some assessment.

import itertools
import os
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Callable, List, Optional, Union

import pandas as pd
from absl import logging
from sklearn import impute
from sklearn.utils.multiclass import type_of_target

import credoai.integration as ci
from credoai.metrics.metrics import find_metrics
from credoai.utils.common import (
    IntegrationError,
    ValidationError,
    flatten_list,
    json_dumps,
    raise_or_warn,
    update_dictionary,
)
from credoai.utils.constants import (
    MODEL_TYPES,
    RISK_ISSUE_MAPPING,
    SUPPORTED_FRAMEWORKS,
)
from credoai.utils.credo_api import CredoApi
from credoai.utils.credo_api_client import CredoApiClient
from credoai.utils.model_utils import get_model_info


class CredoGovernance:
    """Class to store governance data.

    This information is used to interact with the CredoAI
    Governance App.

    At least one of the credo_url or spec_path must be provided! If both
    are provided, the spec_path takes precedence.

    To make use of Governance App a .credo_config file must
    also be set up (see README)

    Parameters
    ----------
    spec_destination: str
        Where to find the assessment spec. Two possibilities. Either:
        * end point to retrieve assessment spec from credo AI's governance platform
        * The file location for the assessment spec json downloaded from
        the assessment requirements of an Use Case on Credo AI's
        Governance App
    warning_level : int
        warning level.
            0: warnings are off
            1: warnings are raised (default)
            2: warnings are raised as exceptions.
    """

    def __init__(self, spec_destination: str = None, warning_level=1):
        self.warning_level = warning_level
        self.assessment_spec = {}
        self.use_case_id = None
        self.model_id = None
        self.dataset_id = None
        self.training_dataset_id = None

        client = CredoApiClient()
        self._api = CredoApi(client=client)

        # set up assessment spec
        if spec_destination:
            self._process_spec(spec_destination)

    def get_assessment_plan(self):
        """Get assessment plan

        Return the assessment plan for the model defined
        by model_id.

        If not retrieved yet, attempt to retrieve the plan first
        from the AI Governance app.
        """
        assessment_plan = defaultdict(dict)
        missing_metrics = []
        for risk_issue, risk_plan in self.assessment_spec.get(
            "assessment_plan", {}
        ).items():
            metrics = [m["type"] for m in risk_plan]
            passed_metrics = []
            for m in metrics:
                found = bool(find_metrics(m))
                if not found:
                    missing_metrics.append(m)
                else:
                    passed_metrics.append(m)
            if risk_issue in RISK_ISSUE_MAPPING:
                update_dictionary(
                    assessment_plan[RISK_ISSUE_MAPPING[risk_issue]],
                    {"metrics": passed_metrics},
                )
        # alert about missing metrics
        for m in missing_metrics:
            logging.warning(
                f"Metric ({m}) is defined in the assessment plan but is not defined by Credo AI.\n"
                "Ensure you create a custom Metric (credoai.metrics.Metric) and add it to the\n"
                "assessment plan passed to lens"
            )
        # remove repeated metrics
        for plan in assessment_plan.values():
            plan["metrics"] = list(set(plan["metrics"]))
        return assessment_plan

    def get_policy_checklist(self):
        return self.assessment_spec.get("policy_questions")

    def get_info(self):
        """Return Credo AI Governance IDs"""
        to_return = self.__dict__.copy()
        del to_return["assessment_spec"]
        del to_return["warning_level"]
        return to_return

    def get_defined_ids(self):
        """Return IDS that have been defined"""
        return [k for k, v in self.get_info().items() if v]

    def register(
        self,
        model_name=None,
        dataset_name=None,
        training_dataset_name=None,
        assessment_template=None,
    ):
        """Registers artifacts to Credo AI Governance App

        Convenience function to register multiple artifacts at once

        Parameters
        ----------
        model_name : str
            name of a model
        dataset_name : str
            name of a dataset used to assess the model
        training_dataset_name : str
            name of a dataset used to train the model
        assessment_template : str
            name of an assessment template that already exists on the governance platform,
            which will be applied to the model

        """
        if model_name:
            self._register_model(model_name)
        if dataset_name:
            self._register_dataset(dataset_name)
        if training_dataset_name:
            self._register_dataset(training_dataset_name, register_as_training=True)
        if assessment_template and self.model_id:
            self._api.apply_assessment_template(
                assessment_template, self.use_case_id, self.model_id
            )
            self._process_spec(
                f"use_cases/{self.use_case_id}/models/{self.model_id}/assessment_spec"
            )

    def export_assessment_results(
        self,
        assessment_results,
        reporter_assets=None,
        destination="credoai",
        assessed_at=None,
    ):
        """Export assessment json to file or credo

        Parameters
        ----------
        assessment_results : dict or list
            dictionary of metrics or
            list of prepared_results from credo_assessments. See lens.export for example
        reporter_assets : list, optional
            list of assets from a CredoReporter, by default None
        destination : str
            Where to send the report
            -- "credoai", a special string to send to Credo AI Governance App.
            -- Any other string, save assessment json to the output_directory indicated by the string.
        assessed_at : str, optional
            date when assessments were created, by default None
        """
        assessed_at = assessed_at or datetime.utcnow().isoformat()
        payload = ci.prepare_assessment_payload(
            assessment_results, reporter_assets=reporter_assets, assessed_at=assessed_at
        )
        if destination == "credoai":
            if self.use_case_id and self.model_id:
                self._api.create_assessment(self.use_case_id, self.model_id, payload)
                logging.info(
                    f"Successfully exported assessments to Credo AI's Governance App"
                )
            else:
                logging.error(
                    "Couldn't upload assessment to Credo AI's Governance App. "
                    "Ensure use_case_id is defined in CredoGovernance"
                )
        else:
            if not os.path.exists(destination):
                os.makedirs(destination, exist_ok=False)
            name_for_save = f"assessment_run-{assessed_at}.json"
            # change name in case of windows
            if os.name == "nt":
                name_for_save = name_for_save.replace(":", "-")
            output_file = os.path.join(destination, name_for_save)
            with open(output_file, "w") as f:
                f.write(json_dumps(payload))

    def set_governance_info_by_name(
        self,
        *,
        use_case_name=None,
        model_name=None,
        dataset_name=None,
        training_dataset_name=None,
    ):
        """Sets governance info by name

        Sets model_id, and/or dataset_id(s)
        using names. This assumes that artifacts have already
        been registered

        Parameters
        ----------
        use_case_name : str
            name of a use_case
        model_name : str
            name of a model
        dataset_name : str
            name of a dataset used to assess the model
        training_dataset_name : str
            name of a dataset used to train the model
        """
        if use_case_name:
            use_case = self._api.get_use_case_by_name(use_case_name)
            if use_case is not None:
                self.use_case_id = use_case["id"]
        if model_name:
            model = self._api.get_model_by_name(model_name)
            if model is not None:
                self.model_id = model["id"]
        if dataset_name:
            dataset = self._api.get_dataset_by_name(dataset_name)
            if dataset is not None:
                self.dataset_id = dataset["id"]
        if training_dataset_name:
            dataset = self._api.get_dataset_by_name(training_dataset_name)
            if dataset is not None:
                self.training_dataset_id = dataset["id"]

    def _process_spec(self, spec_destination):
        self.assessment_spec = ci.process_assessment_spec(spec_destination, self._api)
        self.use_case_id = self.assessment_spec["use_case_id"]
        self.model_id = self.assessment_spec["model_id"]
        self.dataset_id = self.assessment_spec["validation_dataset_id"]
        self.training_dataset_id = self.assessment_spec["training_dataset_id"]

    def _register_dataset(self, dataset_name, register_as_training=False):
        """Registers a dataset

        Parameters
        ----------
        dataset_name : str
            name of a dataset
        register_as_training : bool
            If True and model_id is defined, register dataset to model as training data,
            default False
        """
        prefix = ""
        if register_as_training:
            prefix = "training_"
        try:
            dataset_id = self._api.register_dataset(name=dataset_name)["id"]
            setattr(self, f"{prefix}dataset_id", dataset_id)
        except IntegrationError:
            self.set_governance_info_by_name(**{f"{prefix}dataset_name": dataset_name})
            raise_or_warn(
                IntegrationError,
                f"The dataset ({dataset_name}) is already registered.",
                f"The dataset ({dataset_name}) is already registered. Using registered dataset",
                self.warning_level,
            )
        if not register_as_training and self.model_id and self.use_case_id:
            self._api.register_dataset_to_model_usecase(
                use_case_id=self.use_case_id,
                model_id=self.model_id,
                dataset_id=self.dataset_id,
            )
        if register_as_training and self.model_id and self.training_dataset_id:
            logging.info(
                f"Registering dataset ({dataset_name}) to model ({self.model_id})"
            )
            self._api.register_dataset_to_model(self.model_id, self.training_dataset_id)

    def _register_model(self, model_name):
        """Registers a model

        If a project has not been registered, a new project will be created to
        register the model under.

        If an AI solution has been set, the model will be registered to that
        solution.
        """
        try:
            self.model_id = self._api.register_model(name=model_name)["id"]
        except IntegrationError:
            self.set_governance_info_by_name(model_name=model_name)
            raise_or_warn(
                IntegrationError,
                f"The model ({model_name}) is already registered.",
                f"The model ({model_name}) is already registered. Using registered model",
                self.warning_level,
            )
        if self.use_case_id:
            logging.info(
                f"Registering model ({model_name}) to Use Case ({self.use_case_id})"
            )
            self._api.register_model_to_usecase(self.use_case_id, self.model_id)


class CredoModel:
    """Class wrapper around model-to-be-assessed

    CredoModel serves as an adapter between arbitrary models
    and the assessments in CredoLens. Assessments depend
    on CredoModel instantiating certain methods. In turn,
    the methods an instance of CredoModel defines informs
    Lens which assessment can be automatically run.

    An assessment's required CredoModel functionality can be accessed
    using the `get_requirements` function of an assessment instance.

    The most generic way to interact with CredoModel is to pass a model_config:
    a dictionary where the key/value pairs reflect functions. This method is
    agnostic to framework. As long as the functions serve the needs of the
    assessments, they'll work.

    E.g. {'predict': model.predict}

    The model_config can also be inferred automatically, from well-known packages
    (call CredoModel.supported_frameworks for a list.) If supported, a model
    can be passed directly to CredoModel's "model" argument and a model_config
    will be inferred.

    Note a model or model_config *must* be passed. If both are passed, any
    functionality specified in the model_config will overwrite and inferences
    made from the model itself.

    See the `quickstart notebooks <https://credoai-lens.readthedocs.io/en/stable/notebooks/quickstart.html#CredoModel>`_ for more information about usage


    Parameters
    ----------
    name : str
        Label of the model
    model : model, optional
        A model from a supported framework. Note functionality will be limited
        by CredoModel's automated inference. model_config is a more
        flexible and reliable method of interaction, by default None
    model_config : dict, optional
        dictionary containing mappings between CredoModel function names (e.g., "predict")
        and functions (e.g., "model.predict"), by default None
    model_type : str, optional
        Specifies the type of model. If a model is supplied, CredoModel will attempt to infer
        the model_type (see utils.model_utils.get_model_info).
        When manually input, must be selected from supported MODEL_TYPES
        (Run CredoModel.model_types() to get list of supported types). Set model_type will
        override any inferred type. By default None
    """

    def __init__(
        self, name: str, model=None, model_config: dict = None, model_type: str = None
    ):
        self.name = name
        self.config = {}
        self.model = model
        self.framework = None
        self.model_type = model_type
        assert model is not None or model_config is not None
        if model is not None:
            info = get_model_info(model)
            self.framework = info["framework"]
            self.model_type = info["model_type"]
            self._init_config(model)
        if model_config is not None:
            self.config.update(model_config)
        self._build_functionality()

    @staticmethod
    def model_types():
        return MODEL_TYPES

    @staticmethod
    def supported_frameworks():
        return SUPPORTED_FRAMEWORKS

    def _build_functionality(self):
        for key, val in self.config.items():
            if val is not None:
                self.__dict__[key] = val

    def _init_config(self, model):
        config = {}
        if self.framework == "sklearn":
            config = self._init_sklearn(model)
        elif self.framework == "xgboost":
            config = self._init_xgboost(model)
        self.config = config

    def _init_sklearn(self, model):
        return self._sklearn_style_config(model)

    def _init_xgboost(self, model):
        return self._sklearn_style_config(model)

    def _sklearn_style_config(self, model):
        config = {"predict": model.predict}
        # if binary classification, only return
        # the positive classes probabilities by default
        try:
            if len(model.classes_) == 2:

                def predict_proba(X):
                    return model.predict_proba(X)[:, 1]

            else:
                predict_proba = model.predict_proba

            config["predict_proba"] = predict_proba
        except AttributeError:
            pass
        return config


class CredoData:
    """Class wrapper around data-to-be-assessed

    CredoData serves as an adapter between datasets
    and the assessments in CredoLens.

    Passed to Lens for certain assessments. Either will be used
    by a CredoModel to make predictions or analyzed itself.

    See the `quickstart notebooks <https://credoai-lens.readthedocs.io/en/stable/notebooks/quickstart.html#CredoData>`_ for more information about usage

    Parameters
    -------------
    name : str
        Label of the dataset
    X : array-like of shape (n_samples, n_features)
        Dataset
    y : array-like of shape (n_samples, n_outputs)
        Outcome
    sensitive_features : pd.Series, pd.DataFrame, optional
        Sensitive Features, which will be used for disaggregating performance
        metrics. This can be the columns you want to perform segmentation analysis on, or
        a feature related to fairness like 'race' or 'gender'
    sensitive_intersections : bool, list
        Whether to add intersections of sensitive features. If True, add all possible
        intersections. If list, only create intersections from specified sensitive features.
        If False, no intersections will be created. Defaults False
    categorical_features_keys : list[str], optional
        Names of categorical features. If the sensitive feature is categorical, include it in this list.
        Note - ordinal features should not be included.
    """

    def __init__(
        self,
        name: str,
        X=None,
        y=None,
        sensitive_features=None,
        sensitive_intersections: Union[bool, list] = False,
        categorical_features_keys: Optional[List[str]] = None,
    ):
        self.name = name
        self.X = X
        self.y = y
        self.sensitive_features = self._process_sensitive(
            sensitive_features, sensitive_intersections
        )
        self.X_type = self._get_X_type()
        self.target_type = self._get_y_type()
        self.categorical_features_keys = categorical_features_keys

    def __post_init__(self):
        self.metadata = self.metadata or {}
        self._validate_data()

    def _get_X_type(self):
        return type(self.X)

    def _get_y_type(self):
        return type_of_target(self.y) if self.y else None

    def _process_sensitive(self, sensitive_features, sensitive_intersections):
        df = pd.DataFrame(sensitive_features).copy()
        features = df.columns
        if sensitive_intersections is False or len(features) == 1:
            return df
        elif sensitive_intersections is True:
            sensitive_intersections = features
        intersections = []
        for i in range(2, len(features) + 1):
            intersections += list(itertools.combinations(sensitive_intersections, i))
        for intersection in intersections:
            tmp = df[intersection[0]]
            for col in intersection[1:]:
                tmp = tmp.str.cat(df[col].astype(str), sep="_")
            label = "_".join(intersection)
            df[label] = tmp
        return df

    def _validate_data(self):
        # Validate the types
        if not isinstance(self.sensitive_features, pd.DataFrame):
            raise ValidationError(
                "Sensitive_feature_keys type is "
                + self.sensitive_feature_keys.__class__.__name__
                + " but the required type is pd.DataFrame"
            )
        if self.categorical_features_keys and not isinstance(
            self.categorical_features_keys, list
        ):
            raise ValidationError(
                "Categorical_features_keys type is "
                + self.categorical_features_keys.__class__.__name__
                + " but the required type is list"
            )
        if self.y and len(self.X) != len(self.y):
            raise ValidationError(
                "X and y are not the same length. "
                + f"X Length: {len(self.X)}, y Length: {len(self.y)}"
            )
        if self.sensitive_features and len(self.X) != len(self.sensitive_features):
            raise ValidationError(
                "X and sensitive_features are not the same length. "
                + f"X Length: {len(self.X)}, sensitive_features Length: {len(self.y)}"
            )

        self._validate_X()

    def _validate_X(self):
        # DataFrame validate
        if self.X_type == pd.DataFrame:
            # Validate that the data column names are unique
            if len(self.X.columns) != len(set(self.X.columns)):
                raise ValidationError("X contains duplicate column names")

    def dev_mode(self, frac=0.1):
        """Samples data down for faster assessment and iteration

        Sampling will be stratified across the sensitive feature

        Parameters
        ----------
        frac : float
            The fraction of data to use
        """
        data = self.data.groupby(self.sensitive_features, group_keys=False).apply(
            lambda x: x.sample(frac=frac)
        )
        self._process_data(data)
