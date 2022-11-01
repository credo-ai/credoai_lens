%run training_script.py

# %% [markdown]
# ### Imports

# %%
# Import Lens and necessary artifacts
from credoai.lens import Lens
from credoai.artifacts import ClassificationModel, TabularData

# %% [markdown]
# In lens, the classes that evaluate models and/or datasets are called `evaluators`. In this example we are interested in evaluating the model's fairness. For this we can use the `ModelFairness` evaluator. We'll
# also evaluate the model's performance.

# %%
from credoai.evaluators import ModelFairness, Performance

# %% [markdown]
# ## Lens in 5 minutes

# %% [markdown]
# Below is a basic example where our goal is to evaluate the above model. We'll break down this code [below](#Breaking-Down-The-Steps).
# 
# Briefly, the code is doing four things:
# 
# * Wrapping ML artifacts (like models and data) in Lens objects
# * Initializing an instance of Lens. Lens is the main object that performs evaluations. Under the hood, it creates a `pipeline` of evaluations that are run.
# * Add evaluators to Lens.
# * Run Lens

# %%
# set up model and data artifacts
credo_model = ClassificationModel(name="credit_default_classifier", model_like=model)
credo_data = TabularData(
    name="UCI-credit-default",
    X=X_test,
    y=y_test,
    sensitive_features=sensitive_features_test,
)

# Initialization of the Lens object
lens = Lens(model=credo_model, assessment_data=credo_data)

# initialize the evaluator and add it to Lens
metrics = ['precision_score', 'recall_score', 'equal_opportunity']
lens.add(ModelFairness(metrics=metrics), id='MyModelFairness')
lens.add(Performance(metrics=metrics), id='MyModelPerformance')

# run Lens
lens.run()
lens.get_results()['MyModelFairness'][0]

# %% [markdown]
# `lens.get_results()` provides a dictionary where the results of the evaluators are stored as values, and the keys correspond to the ids of the evaluators.  
# 
# In the previous case we specified the id of the evaluator when we added `ModelFairness` to the pipeline, however `id` is an optional argument for the `add` method. If omitted, a random one will be generated.

# %% [markdown]
# ## Using Len's pipeline argument

# %% [markdown]
# If we want to add multiple evaluators to our pipeline, one way of doing it could be repeating the `add` step, as shown above. Another way is to define the pipeline steps, and pass it to `Lens` at initialization time. Let's explore the latter!

# %%
pipeline = [
    (ModelFairness(metrics), "MyModelFairness"),
    (Performance(metrics), "MyModelPerformance"),
]
lens = Lens(model=credo_model, assessment_data=credo_data, pipeline=pipeline)

# %% [markdown]
# Above, each of the `tuples` in the `list` is in the form `(instantiated_evaluator, id)`.

# %%
# notice that Lens functions can be chained together
results = lens.run().get_results()

# %% [markdown]
# Let's check that we have results for both of our evaluators.

# %%
results['MyModelFairness'][0]

# %%
results['MyModelPerformance'][0]

# %% [markdown]
# ## That's it!
# 
# That should get you up and running. Next steps include:
# 
# * Trying out other evaluators (they are all accessible via `credoai.evaluators`)
# * Checking out our developer guide to better understand the Lens ecosystem and see how you can extend it.
# * Exploring the Credo AI Governance Platform, which will connect AI assessments with customizable governance to support reporting, compliance, multi-stakeholder translation and more!

# %% [markdown]
# ## Breaking Down The Steps
# 
# ### Preparing artifacts
# 
# Lens interacts with "AI Artifacts" which wrap model and data objects and standardize them for use by different evaluators.
# 
# Below we create a `ClassificationModel` artifact. This is a light wrapper for any kind of fitted classification model-like object. 
# 
# We also create a `TabularData` artifact which stores X, y and sensitive features.

# %%
# set up model and data artifacts
credo_model = ClassificationModel(name="credit_default_classifier", model_like=model)

credo_data = TabularData(
    name="UCI-credit-default",
    X=X_test,
    y=y_test,
    sensitive_features=sensitive_features_test,
)

# %% [markdown]
# #### Model
# 
# Model type objects, like `ClassificationModel` used above, serve as adapters between arbitrary models and the evaluators in Lens. Some evaluators depend on Model instantiating certain methods. For example, `ClassificationModel` can accept any generic object having `predict` and `predict_proba` methods, including fitted sklear pipelines.
# 
# 

# %% [markdown]
# #### Data
# 
# _Data_ type artifact, like `TabularData` serve as adapters between datasets and the evaluators in Lens.
# 
# When you pass data to a _Data_ artifact, the artifact performs various steps of validation, and formats them so that they can be used by evaluators. The aim of this procedure is to preempt errors down the line.
# 
# You can pass Data to Lens as a **training dataset** or an **assessment dataset** (see lens class documentation). If the former, it will not be used to assess the model. Instead, dataset assessments will be performed on the dataset (e.g., fairness assessment). The validation dataset will be assessed in the same way, but _also_ used to assess the model, if provided.
# 
# Similarly to _Model_ type objects, _Data_ objects can be customized, see !!insertlink!!

# %% [markdown]
# ### Evaluators 
# 
# Lens uses the above artifacts to ensure a successfull run of the evaluators. As we have seen in the sections [Lens in 5 minutes](##Lens-in-5-minutes) and [Adding multiple evaluators](##Adding-multiple-evaluators), multiple evaluators can be added to _Lens_ pipeline. Each evaluators contains information on what it needs in order to run successfully, and it executes a validation step at _add_ time.
# 
# The result of the validation depends on what artifacts are available, their content and the type of evaluator being added to the pipeline. In case the validation process fails, the user is notified the reason why the evaluator cannot be added to the pipeline.
# 
# See for example:

# %%
from credoai.evaluators import Privacy
lens.add(Privacy())

# %% [markdown]
# Currently no automatic run of evaluators is supported. However, when Lens is used in combination with Credo AI Platform, it is possible to download an assessment plan which then gets converted into a set of evaluations that Lens can run programmatically. For more information: !!insert link!!

# %% [markdown]
# ### Run Lens
# 
# After we have initialized _Lens_ the _Model_ and _Data_ (`ClassificationModel` and `TabularData` in our example) type artifacts, we can add whichever evaluators we want to the pipeline, and finally run it!

# %%
lens = Lens(model=credo_model, assessment_data=credo_data)
metrics = ['precision_score', 'recall_score', 'equal_opportunity']
lens.add(ModelFairness(metrics=metrics), id='MyModelFairness')
lens.run()

# %% [markdown]
# As you can notice, when adding _evaluators_ to lens, they need to be instantiated. If any extra arguments need to be passed to the evaluator (like metrics in this case), this is the time to do it.

# %% [markdown]
# **Getting Evaluator Results**
# 
# Afte the pipeline is run, the results become accessible via the method `get_results()`
# 
# `lens.get_results()` provides a dictionary where the results of the evaluators are stored as values, and the keys correspond to the ids of the evaluators.  
# 
# In the previous case we specified the id of the evaluator when we added `ModelFairness` to the pipeline, however `id` is an optional argument for the `add` method. If omitted, a random one will be generated.

# %%
lens.get_results()

# %% [markdown]
# **Credo AI Governance Platform**
# 
# For information on how to interact with the plaform, please look into: [Connecting with Governance App](https://credoai-lens.readthedocs.io/en/stable/notebooks/governance_integration.html) tutorial for directions.
# 


