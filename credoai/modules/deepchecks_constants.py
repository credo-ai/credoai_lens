from deepchecks.checks import *

DEFAULT_CHECKS = [
    ConfusionMatrixReport(),
    SegmentPerformance(),
    SimpleModelComparison(),
    CalibrationScore(),
    RegressionSystematicError(),
    RegressionErrorDistribution(),
    UnusedFeatures(),
    BoostingOverfit(),
    ModelInferenceTime(),
]
