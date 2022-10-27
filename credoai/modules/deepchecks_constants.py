from deepchecks.tabular.checks import *

DEFAULT_CHECKS = [
    ConfusionMatrixReport(),
    WeakSegmentsPerformance(),
    SimpleModelComparison(),
    CalibrationScore(),
    RegressionSystematicError(),
    RegressionErrorDistribution(),
    UnusedFeatures(),
    BoostingOverfit(),
    ModelInferenceTime(),
]
