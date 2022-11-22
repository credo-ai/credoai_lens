
Shap Explainer
==============


This evaluator perform the calculation of shapley values for a dataset/model,
leveraging the SHAP package.

It supports 2 types of assessments:

1. Overall statistics of the shap values across all samples: mean and mean(|x|)
2. Individual shapley values for a list of samples

Sampling
--------
In order to speed up computation time, at the stage in which the SHAP explainer is
initialized, a down sampled version of the dataset is passed to the `Explainer`
object as background data. This is only affecting the calculation of the reference
value, the calculation of the shap values is still performed on the full dataset.

Two strategies for down sampling are provided:

1. Random sampling (the default strategy): the amount of samples can be specified
   by the user.
2. Kmeans: summarizes a dataset with k mean centroids, weighted by the number of
   data points they each represent. The amount of centroids can also be specified
   by the user.

There is no consensus on the optimal down sampling approach. For reference, see this
conversation: https://github.com/slundberg/shap/issues/1018


Categorical variables
---------------------
The interpretation of the results for categorical variables can be more challenging, and
dependent on the type of encoding utilized. Ordinal or one/hot encoding can be hard to
interpret.

There is no agreement as to what is the best strategy as far as categorical variables are
concerned. A good discussion on this can be found here: https://github.com/slundberg/shap/issues/451

No restriction on feature type is imposed by the evaluator, so user discretion in the
interpretation of shap values for categorical variables is advised.


Parameters
----------
samples_ind : Optional[List[int]], optional
    List of row numbers representing the samples for which to extract individual
    shapley values. This must be a list of integer indices. The underlying SHAP
    library does not support non-integer indexing.
background_samples: int,
    Amount of samples to be taken from the dataset in order to build the reference values.
    See documentation about sampling above. Unused if background_kmeans is not False.
background_kmeans : Union[bool, int], optional
    If True, use SHAP kmeans to create a data summary to serve as background data for the
    SHAP explainer using 50 centroids by default. If an int is provided,
    that will be used as the number of centroids. If False, random sampling will take place.


