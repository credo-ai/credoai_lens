from itertools import product

import pandas as pd
from credoai.modules.stats_utils import columns_from_formula
from lifelines import CoxPHFitter


class CoxPH:
    def __init__(self, **kwargs):
        self.name = "Cox Proportional Hazard"
        self.cph = CoxPHFitter(**kwargs)
        self.fit_kwargs = {}
        self.data = None

    def fit(self, data, **fit_kwargs):
        self.cph.fit(data, **fit_kwargs)
        self.fit_kwargs = fit_kwargs
        self.data = data
        if "formula" in fit_kwargs:
            self.name += f" (formula: {fit_kwargs['formula']})"
        return self

    def summary(self):
        s = self.cph.summary
        s.name = f"{self.name} Stat Summary"
        return s

    def expected_survival(self):
        prediction_data = self._get_prediction_data()
        expected_predictions = self.cph.predict_expectation(prediction_data)
        expected_predictions.name = "E(time survive)"
        final = pd.concat([prediction_data, expected_predictions], axis=1)
        final.name = f"{self.name} Expected Survival"
        return final

    def survival_curves(self):
        prediction_data = self._get_prediction_data()
        survival_curves = self.cph.predict_survival_function(prediction_data)
        survival_curves = (
            # fmt: off
            survival_curves.loc[0:,]
            # fmt: on
            .rename_axis("time_step")
            .reset_index()
            .melt(id_vars=["time_step"])
            .merge(right=prediction_data, left_on="variable", right_index=True)
            .drop(columns=["variable"])
        )
        survival_curves = survival_curves[survival_curves["time_step"] % 5 == 0]
        survival_curves.name = f"{self.name} Survival Curves"
        return survival_curves

    def _get_prediction_data(self):
        columns = columns_from_formula(self.fit_kwargs.get("formula"))
        df = pd.DataFrame(
            list(product(*[i.unique() for _, i in self.data[columns].iteritems()])),
            columns=columns,
        )
        return df
