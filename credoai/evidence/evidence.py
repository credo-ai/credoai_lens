import pandas as pd


@pd.api.extensions.register_dataframe_accessor("metric")
class MetricAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # verify there is a column latitude and a column longitude
        if "label" not in obj.columns or "value" not in obj.columns:
            raise AttributeError("Must have 'label' and 'value' columns.")
