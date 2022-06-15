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

import os
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Callable, List, Optional, Union

import pandas as pd
# This file defines CredoGovernance, CredoModel and CredoData
from absl import logging
from sklearn.impute import SimpleImputer
from sklearn.utils.multiclass import type_of_target

import credoai.integration as ci
from credoai.metrics.metrics import find_metrics
from credoai.utils.common import (IntegrationError, ValidationError,
                                  flatten_list, json_dumps, raise_or_warn,
                                  update_dictionary)
from credoai.utils.constants import (MODEL_TYPES, RISK_ISSUE_MAPPING,
                                     SUPPORTED_FRAMEWORKS)
from credoai.utils.credo_api_utils import (get_dataset_by_name,
                                           get_dataset_name, get_model_by_name,
                                           get_model_name,
                                           get_use_case_by_name,
                                           get_use_case_name)
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

    def __init__(self,
                 spec_destination: str = None,
                 warning_level=1):
        self.warning_level = warning_level
        self.assessment_spec = {}
        self.use_case_id = None
        self.model_id = None
        self.dataset_id = None
        self.training_dataset_id = None
        # set up assessment spec
        if spec_destination:
            self.assessment_spec = ci.process_assessment_spec(spec_destination)
            self.use_case_id = self.assessment_spec['use_case_id']
            self.model_id = self.assessment_spec['model_id']
            self.dataset_id = self.assessment_spec['validation_dataset_id']
            self.training_dataset_id = self.assessment_spec['training_dataset_id']

    def get_assessment_plan(self):
        """Get assessment plan

        Return the assessment plan for the model defined
        by model_id.

        If not retrieved yet, attempt to retrieve the plan first
        from the AI Governance app. 
        """
        assessment_plan = defaultdict(dict)
        missing_metrics = []
        for risk_issue, risk_plan in self.assessment_spec.get('assessment_plan', {}).items():
            metrics = [m['type'] for m in risk_plan]
            passed_metrics = []
            for m in metrics:
                found = bool(find_metrics(m))
                if not found:
                    missing_metrics.append(m)
                else:
                    passed_metrics.append(m)
            if risk_issue in RISK_ISSUE_MAPPING:
                update_dictionary(assessment_plan[RISK_ISSUE_MAPPING[risk_issue]],
                                  {'metrics': passed_metrics})
        # alert about missing metrics
        for m in missing_metrics:
            logging.warning(f"Metric ({m}) is defined in the assessment plan but is not defined by Credo AI.\n"
                            "Ensure you create a custom Metric (credoai.metrics.Metric) and add it to the\n"
                            "assessment plan passed to lens")
        # remove repeated metrics
        for plan in assessment_plan.values():
            plan['metrics'] = list(set(plan['metrics']))
        return assessment_plan

    def get_policy_checklist(self):
        return self.assessment_spec.get('policy_questions')

    def get_info(self):
        """Return Credo AI Governance IDs"""
        to_return = self.__dict__.copy()
        del to_return['assessment_spec']
        del to_return['warning_level']
        return to_return

    def get_defined_ids(self):
        """Return IDS that have been defined"""
        return [k for k, v in self.get_info().items() if v]

    def register(self,
                 model_name=None,
                 dataset_name=None,
                 training_dataset_name=None):
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

        """
        if model_name:
            self._register_model(model_name)
        if dataset_name:
            self._register_dataset(dataset_name)
        if training_dataset_name:
            self._register_dataset(training_dataset_name,
                                   register_as_training=True)

    def export_assessment_results(self,
                                  assessment_results,
                                  reporter_assets=None,
                                  destination='credoai',
                                  assessed_at=None
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
        assessed_at = assessed_at or datetime.now().isoformat()
        payload = ci.prepare_assessment_payload(
            assessment_results, reporter_assets=reporter_assets, assessed_at=assessed_at)
        if destination == 'credoai':
            if self.use_case_id and self.model_id:
                ci.post_assessment(self.use_case_id,
                                   self.model_id, payload)
                logging.info(
                    f"Exporting assessments to Credo AI's Governance App")
            else:
                logging.error("Couldn't upload assessment to Credo AI's Governance App. "
                              "Ensure use_case_id is defined in CredoGovernance")
        else:
            if not os.path.exists(destination):
                os.makedirs(destination, exist_ok=False)
            name_for_save = f"assessment_run-{assessed_at}.json"
            # change name in case of windows
            if os.name == 'nt':
                name_for_save = name_for_save.replace(':', '-')
            output_file = os.path.join(destination, name_for_save)
            with open(output_file, 'w') as f:
                f.write(json_dumps(payload))

    def set_governance_info_by_name(self,
                                    *,
                                    use_case_name=None,
                                    model_name=None,
                                    dataset_name=None,
                                    training_dataset_name=None):
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
            ids = get_use_case_by_name(use_case_name)
            if ids is not None:
                self.use_case_id = ids['use_case_id']
        if model_name:
            ids = get_model_by_name(model_name)
            if ids is not None:
                self.model_id = ids['model_id']
        if dataset_name:
            ids = get_dataset_by_name(dataset_name)
            if ids is not None:
                self.dataset_id = ids['dataset_id']
        if training_dataset_name:
            ids = get_dataset_by_name(training_dataset_name)
            if ids is not None:
                self.training_dataset_id = ids['dataset_id']

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
        prefix = ''
        if register_as_training:
            prefix = 'training_'
        try:
            ids = ci.register_dataset(dataset_name=dataset_name)
            setattr(self, f'{prefix}dataset_id', ids['dataset_id'])
        except IntegrationError:
            self.set_governance_info_by_name(
                **{f'{prefix}dataset_name': dataset_name})
            raise_or_warn(IntegrationError,
                          f"The dataset ({dataset_name}) is already registered.",
                          f"The dataset ({dataset_name}) is already registered. Using registered dataset",
                          self.warning_level)
        if not register_as_training and self.model_id and self.use_case_id:
            ci.register_dataset_to_model_usecase(
                use_case_id=self.use_case_id, model_id=self.model_id, dataset_id=self.dataset_id
            )
        if register_as_training and self.model_id and self.training_dataset_id:
            logging.info(
                f"Registering dataset ({dataset_name}) to model ({self.model_id})")
            ci.register_dataset_to_model(
                self.model_id, self.training_dataset_id)

    def _register_model(self, model_name):
        """Registers a model

        If a project has not been registered, a new project will be created to
        register the model under.

        If an AI solution has been set, the model will be registered to that
        solution.
        """
        try:
            ids = ci.register_model(model_name=model_name)
            self.model_id = ids['model_id']
        except IntegrationError:
            self.set_governance_info_by_name(model_name=model_name)
            raise_or_warn(IntegrationError,
                          f"The model ({model_name}) is already registered.",
                          f"The model ({model_name}) is already registered. Using registered model",
                          self.warning_level)
        if self.use_case_id:
            logging.info(
                f"Registering model ({model_name}) to Use Case ({self.use_case_id})")
            ci.register_model_to_usecase(self.use_case_id, self.model_id)


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
        self,
        name: str,
        model=None,
        model_config: dict = None,
        model_type: str = None
    ):
        self.name = name
        self.config = {}
        self.model = model
        self.framework = None
        assert model is not None or model_config is not None
        if model is not None:
            info = get_model_info(model)
            self.framework = info['framework']
            self.model_type = info['model_type']
            self._init_config(model)
        if model_type:
            self.model_type = model_type
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
        if self.framework == 'sklearn':
            config = self._init_sklearn(model)
        elif self.framework == 'xgboost':
            config = self._init_xgboost(model)
        self.config = config

    def _init_sklearn(self, model):
        return self._sklearn_style_config(model)

    def _init_xgboost(self, model):
        return self._sklearn_style_config(model)

    def _sklearn_style_config(self, model):
        config = {'predict': model.predict}
        # if binary classification, only return
        # the positive classes probabilities by default
        try:
            if len(model.classes_) == 2:
                def predict_proba(X): return model.predict_proba(X)[:, 1]
            else:
                predict_proba = model.predict_proba

            config['predict_proba'] = predict_proba
        except AttributeError:
            pass
        return config


class CredoData:
    """Class wrapper around data-to-be-assessed

    CredoData serves as an adapter between tabular datasets
    and the assessments in CredoLens.

    Passed to Lens for certain assessments. Either will be used
    by a CredoModel to make predictions or analyzed itself. 

    See the `quickstart notebooks <https://credoai-lens.readthedocs.io/en/stable/notebooks/quickstart.html#CredoData>`_ for more information about usage

    Parameters
    -------------
    name : str
        Label of the dataset
    data : pd.DataFrame
        Dataset dataframe that includes all features and labels
    label_key : str
        Name of the label column
    sensitive_feature_key : str, optional
        Name of the sensitive feature column, which will be used for disaggregating performance
        metrics. This can a column you want to perform segmentation analysis on, or
        a feature related to fairness like 'race' or 'gender'
    categorical_features_keys : list[str], optional
        Names of categorical features. If the sensitive feature is categorical, include it in this list.
        Note - ordinal features should not be included. 
    unused_features_keys : list[str], optional
        Names of the features to ignore when performing prediction.
        Include all the features in the data that were not used during model training
    drop_sensitive_feature : bool, optional
        If True, automatically adds sensitive_feature_key to the list of 
        unused_features_keys. If you do not explicitly use the sensitive feature
        in your model, this argument should be True. Otherwise, set to False.
        Default, True
    nan_strategy : str or callable, optional
        The strategy for dealing with NaNs when get_scrubbed_data is called. Note, only some
        assessments used the scrubbed data. In general, recommend you deal with NaNs before
        passing your data to Lens. 

        -- If "ignore" do nothing,
        -- If "drop" drop any rows with any NaNs. 
        -- If any other string, pass to the "strategy" argument of `Simple Imputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html>`_.

        You can also supply your own imputer with
        the same API as `SimpleImputer <https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html>`_.
    """

    def __init__(self,
                 name: str,
                 data: pd.DataFrame,
                 label_key: str,
                 sensitive_feature_keys: list = None,
                 categorical_features_keys: Optional[List[str]] = None,
                 unused_features_keys: Optional[List[str]] = None,
                 drop_sensitive_feature: bool = True,
                 nan_strategy: Union[str, Callable] = 'drop'
                 ):

        self.name = name
        self.data = data
        self.sensitive_feature_keys = sensitive_feature_keys
        self.label_key = label_key
        self.categorical_features_keys = categorical_features_keys
        self.unused_features_keys = unused_features_keys
        self.drop_sensitive_feature = drop_sensitive_feature
        self.nan_strategy = nan_strategy
        self.X, self.y, self.sensitive_features = self._process_data(
            self.data).values()
        self.target_type = type_of_target(self.y)

    def __post_init__(self):
        self.metadata = self.metadata or {}
        self._validate_data()

    def get_scrubbed_data(self):
        """Return scrubbed data

        Implements NaN strategy indicated by nan_strategy before returning
        X, y and sensitive_features dataframes/series.

        Returns
        -------
        pd.DataFrame, pd.pd.Series
            X, y, sensitive_features

        Raises
        ------
        ValueError
            ValueError raised for nan_strategy cannot be used by SimpleImputer
        """
        data = self.data.copy()
        if self.nan_strategy == 'drop':
            data = data.dropna()
        elif self.nan_strategy == 'ignore':
            pass
        elif isinstance(self.nan_strategy, str):
            try:
                imputer = SimpleImputer(strategy=self.nan_strategy)
                imputed = imputer.fit_transform(data)
                data.iloc[:, :] = imputed
            except ValueError:
                raise ValueError(
                    "CredoData's nan_strategy could not be successfully passed to SimpleImputer as a 'strategy' argument")
        else:
            imputed = self.nan_strategy.fit_transform(data)
            data.iloc[:, :] = imputed
        return self._process_data(data)

    def _process_data(self, data):
        # set up sensitive features, y and X
        y = data[self.label_key]
        to_drop = [self.label_key]
        if self.unused_features_keys:
            to_drop += self.unused_features_keys

        sensitive_features = None
        if self.sensitive_feature_keys:
            sensitive_features = data[self.sensitive_feature_keys]
            if self.drop_sensitive_feature:
                to_drop.extend(self.sensitive_feature_keys)

        # drop columns from X
        X = data.drop(columns=to_drop, axis=1)
        return {'X': X, 'y': y, 'sensitive_features': sensitive_features}

    def _validate_data(self):
        # Validate the types
        if not isinstance(self.data, pd.DataFrame):
            raise ValidationError(
                "The provided data type is " + self.data.__class__.__name__ +
                " but the required type is pd.DataFrame"
            )
        if not isinstance(self.sensitive_feature_keys, list):
            raise ValidationError(
                "The provided sensitive_feature_keys type is " +
                self.sensitive_feature_keys.__class__.__name__ + " but the required type is list"
            )
        if not isinstance(self.label_key, str):
            raise ValidationError(
                "The provided label_key type is " +
                self.label_key.__class__.__name__ + " but the required type is str"
            )
        if self.categorical_features_keys and not isinstance(self.categorical_features_keys, list):
            raise ValidationError(
                "The provided label_key type is " +
                self.label_key.__class__.__name__ + " but the required type is list"
            )
        # Validate that the data column names are unique
        if len(self.data. columns) != len(set(self.data. columns)):
            raise ValidationError(
                "The provided data contains duplicate column names"
            )
        # Validate that the data contains the provided sensitive feature and label keys
        col_names = list(self.data.columns)
        for sensitive_feature_key in self.sensitive_feature_keys:
            if sensitive_feature_key not in col_names:
                raise ValidationError(
                    "The provided sensitive_feature_key " + sensitive_feature_key +
                    " does not exist in the provided data"
                )
        if self.label_key not in col_names:
            raise ValidationError(
                "The provided label_key " + self.label_key +
                " does not exist in the provided data"
            )

    def dev_mode(self, frac=0.1):
        """Samples data down for faster assessment and iteration

        Sampling will be stratified across the sensitive feature

        Parameters
        ----------
        frac : float
            The fraction of data to use
        """
        data = self.data.groupby(self.sensitive_features,
                                 group_keys=False).apply(lambda x: x.sample(frac=frac))
        self._process_data(data)
