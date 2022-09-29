
Modules & Assessments
---------------------
Modules are a broad class. They can be anything - any tool you'd want to run on a model
or dataset. While Credo AI has defined some modules of our own, your own code can be 
thought of as a module (inherit from the abstract `CredoModule <https://github.com/credo-ai/credoai_lens/blob/develop/credoai/modules/credo_module.py>`_ class), as could other tools available in the broader AI ecosystem.

Some well known packages already serve as the foudnation for some Lens modules, such as:

* Fairlearn
* Adversarial Robustness Toolbox
* Pandas Profiler

Because the class of modules is  unconstrained, we need a way to standardize
their API. We do that in the form of CredoAssessments. CredoAssessments are 
wrappers around one or more modules that allow them to connect to 
CredoModels and Data (:ref:`Credo Artifacts: Model, Data, & Governance`).

Assessments have certain functionality requirements, which the CredoModel/Data must meet to be run.
Essentially, we use "duck typing" for models. Assessments require certain functionality and can
run on any object that initiates that functionality. Lens makes use of these functionality requirements
to automatically determine which assessments to run.

You can easily define your own assessment by inheriting from the abstract `CredoAssessment <https://github.com/credo-ai/credoai_lens/blob/develop/credoai/assessment/credo_assessment.py>`_ class.

.. image:: _static/images/assessments_modules_schematic.png
   :width: 600


Credo Artifacts: Model, Data, & Governance
-------------------------------------------
AI Models and Datasets take many forms. This flexibility has many benefits, but is
an obstacle when your goal is to connect any model or dataset to any assessment! To
solve this issue we introduce two artifacts: CredoModels and CredoData.

**Credo Models** are not "models" in the traditional sense - they are connector objects
that instantiate functions necessary for an assessment. For instance, to evaluate
fairness using the "Fairness" assessment, the CredoModel must instantiate
a `predict_proba` or `predict`. The nature of these functions can be quite general.

The simplest case is you setting CredoModel's `predict` to the `predict` method of your model.
But your "model" may actually be an API call that you want to assess, in which case
the `predict` may be an API call.

Some functions can be inferred from well-known frameworks like scikit-learn. This allows
the CredoModel to be automatically set up, though further customization is possible.

**Credo Data** are simple objects that normalize datasets.
Data assessments are run on these.

**CredoGovernance** is the connection between Lens and the Governance App. This is only relevant
for that use-case.


The Assessment Plan
------------------
An "Assessment Plan" must be supplied to Lens. The plan configures
how different assessments should be run.

The plan makes most sense as part of Credo AI's overall governance app. In 
this context the Assessment Plan is the output of a multi-stakeholder articulation of
how the AI system should be assessed. Lens actually takes care of automatically
retrieving the Assessment Plan from Governance App, connecting 
your technical team to compliance, product, and other stakeholders.

Of course, a single person can also defined the Plan. 
In this case, the Assessment Plan still serves as an initial decision
as to how assessments should be run, and summary of the assessments run.

Beyond the Assessment Plan, each module has other parameters that can be configured. 
See :ref:`the FAQ <lens faq>` for more information.


Credo AI Governance App
----------------------------
Assessment is important, but it's not the end all of Responsible AI development!
`Credo AI's <https://www.credo.ai/>`_ Governance App provides the other aspects needed for effective
AI governance including: support for multi-stakeholder alignment, policy packs
for different areas of responsible AI development and compliance needs,
and continuous translation of all evidence (including assessment result!) into
a risk perspective.

This app is independent from using Lens for assessments. You can use *any*
method to assess your AI artifacts and upload the results to the Governance App
smoothly. Check out the `integration demo <https://credoai-lens.readthedocs.io/en/latest/notebooks/integration_demo.html>`_ to see how that is done.

