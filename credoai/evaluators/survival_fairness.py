from credoai.artifacts import TabularData
from credoai.evaluators import Evaluator
from credoai.evaluators.utils.validation import (
    check_artifact_for_nulls,
    check_data_instance,
    check_existence,
)
from credoai.evidence import TableContainer
from credoai.modules import CoxPH
from credoai.modules.stats_utils import columns_from_formula
from credoai.utils import ValidationError


class SurvivalFairness(Evaluator):
    def __init__(self, CoxPh_kwargs=None, confounds=None):
        if CoxPh_kwargs is None:
            CoxPh_kwargs = {"duration_col": "duration", "event_col": "event"}
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
        if "formula" in self.coxPh_kwargs:
            cph = CoxPH()
            cph.fit(self.survival_df, **self.coxPh_kwargs)
            self.stats.append(cph)
            return

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
        if expected_columns is not None:
            missing_columns = expected_columns.difference(self.assessment_data.X)
            if missing_columns:
                raise ValidationError(
                    f"Columns supplied to CoxPh formula not found in data. Columns are: {missing_columns}"
                )

    def _get_expected_survival(self):
        return [s.expected_survival() for s in self.stats]

    def _get_summaries(self):
        return [s.summary() for s in self.stats]

    def _get_survival_curves(self):
        return [s.survival_curves() for s in self.stats]
