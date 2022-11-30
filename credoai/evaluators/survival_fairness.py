from connect.evidence import TableContainer

from credoai.artifacts import TabularData
from credoai.evaluators import Evaluator
from credoai.evaluators.utils.validation import (
    check_data_instance,
    check_existence,
    check_features_presence,
)
from credoai.modules import CoxPH
from credoai.modules.stats_utils import columns_from_formula


class SurvivalFairness(Evaluator):
    """Performs survival analysis on a dataset and compares it to model predictions

    This evaluator uses a Cox Proportional Hazard model to analyze
    the "survival" in a dataset as a function of sensitive features. That is, is there a
    relationship between survival rate and the sensitive category. In addition, an arbitrary
    formula can be passed to run a custom CoxPH model.

    This evaluator provides functionality to:

    - calculate expected survival
    - create survival curves
    - calculate beta coefficients for sensitive features affected survival

    Parameters
    ----------
    CoxPh_kwargs : dict, optional
        arguments to pass to the `fit` function of lifelines.CoxPHFiter, by default None
    confounds : list, optional
        List of features in assessment_data.X to include in survival model
        as confounds, by default None
    """

    def __init__(self, CoxPh_kwargs=None, confounds=None):
        if CoxPh_kwargs is None:
            CoxPh_kwargs = {}
        self.coxPh_kwargs = CoxPh_kwargs
        self.confounds = confounds
        self.stats = []

    required_artifacts = ["model", "assessment_data", "sensitive_feature"]

    def evaluate(self):
        self._run_survival_analyses()
        result_dfs = (
            self._get_summaries()
            + self._get_expected_survival()
            + self._get_survival_curves()
        )
        sens_feat_label = {"sensitive_feature": self.sensitive_name}
        self.results = [
            TableContainer(df, **self.get_container_info(labels=sens_feat_label))
            for df in result_dfs
        ]
        return self

    def _setup(self):
        self.y_pred = self.model.predict(self.assessment_data.X)
        self.sensitive_name = self.assessment_data.sensitive_feature.name
        self.survival_df = self.assessment_data.X.copy()
        self.survival_df["predictions"] = self.y_pred
        self.survival_df = self.survival_df.join(self.assessment_data.sensitive_feature)
        # add probabilities
        try:
            self.y_prob = self.model.predict_proba(self.assessment_data.X)
            self.survival_df["predicted_probabilities"] = self.y_prob
        except:
            self.y_prob = None
        return self

    def _run_survival_analyses(self):
        # if formula is provided, also run that analysis
        if "formula" in self.coxPh_kwargs:
            cph = CoxPH()
            cph.fit(self.survival_df, **self.coxPh_kwargs)
            self.stats.append(cph)

        model_predictions = (
            ["predictions", "predicted_probabilities"]
            if self.y_prob is not None
            else ["predictions"]
        )
        for pred in model_predictions:
            run_kwargs = self.coxPh_kwargs.copy()
            run_kwargs["formula"] = f"{self.sensitive_name} * {pred}"
            if self.confounds:
                run_kwargs["formula"] += " + ".join(["", *self.confounds])
            cph = CoxPH()
            cph.fit(self.survival_df, **run_kwargs)
            self.stats.append(cph)

    def _get_expected_survival(self):
        return [s.expected_survival() for s in self.stats]

    def _get_summaries(self):
        return [s.summary() for s in self.stats]

    def _get_survival_curves(self):
        return [s.survival_curves() for s in self.stats]

    def _validate_arguments(self):
        check_data_instance(self.assessment_data, TabularData)
        check_existence(self.assessment_data.sensitive_features, "sensitive_features")
        # check for columns existences
        expected_columns = None
        if self.confounds:
            expected_columns = set(self.confounds)
        if "formula" in self.coxPh_kwargs:
            expected_columns |= columns_from_formula(self.coxPh_kwargs["formula"])
            expected_columns -= {"predictions", "predicted_probabilities"}
            expected_columns.add(self.coxPh_kwargs.get("duration_col", "duration_col"))
            expected_columns.add(self.coxPh_kwargs.get("event_col", "event_col"))
        if expected_columns is not None:
            check_features_presence(expected_columns)
