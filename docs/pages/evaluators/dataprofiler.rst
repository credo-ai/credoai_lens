
Data Profiler
=============


Data profiling module for Credo AI.

This evaluator runs the pandas profiler on a data. Pandas profiler calculates a number
of descriptive statistics about the data.

Parameters
----------
X : pandas.DataFrame
    The features
y : pandas.Series
    The outcome labels
profile_kwargs
    Passed to pandas_profiling.ProfileReport
