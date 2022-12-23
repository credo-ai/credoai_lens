"""
Containers for different types of data and models

In order to ensure Lens runs correctly across a variety of data formats
and model types, user's inputs are wrapped into classes that format/validate
them.
"""
from .data.base_data import Data
from .data.comparison_data import ComparisonData
from .data.tabular_data import TabularData
from .model.base_model import Model
from .model.classification_model import ClassificationModel, DummyClassifier
from .model.comparison_model import ComparisonModel, DummyComparisonModel
from .model.regression_model import DummyRegression, RegressionModel
