
Data Profiler (Experimental)
============================


Data profiling evaluator for Credo AI (Experimental)

This evaluator runs the pandas profiler on a data. Pandas profiler calculates a number
of descriptive statistics about the data.

Parameters
----------
dataset_name: str
    Name of the dataset
profile_kwargs
    Potential arguments to be passed to pandas_profiling.ProfileReport
