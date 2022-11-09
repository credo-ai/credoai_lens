from deepchecks.tabular import checks

DEFAULT_CHECKS = [
    checks.ConfusionMatrixReport(),
    checks.WeakSegmentsPerformance(),
    checks.SimpleModelComparison(),
    checks.CalibrationScore(),
    checks.RegressionSystematicError(),
    checks.RegressionErrorDistribution(),
    checks.UnusedFeatures(),
    checks.BoostingOverfit(),
    checks.ModelInferenceTime(),
]
