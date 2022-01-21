..
   Note: Items in this toctree form the top-level navigation. See `api.rst` for the `autosummary` directive, and for why `api.rst` isn't called directly.

.. toctree::
   :hidden:

   Home page <self>
   Setup <setup>
   Jupyter tutorials <tutorials>
   API reference <_autosummary/credoai>

Lens by Credo AI
===============

Lens is Credo AI's AI Assessment Framework. It supports assessment
of AI systems, with a focus on responsible AI assessment. It also
streamlines integration with the Credo AI governance platform.

Check on the `quickstart tutorial <https://credoai-lens.readthedocs.io/en/latest/notebooks/quickstart.html>`_
to get started.

If you are connecting to the Credo AI Governance platform, see the `governance integration tutorial <https://credoai-lens.readthedocs.io/en/latest/notebooks/governance_integration.html>`_.

Overview
--------
Lens is made of a few components.

* **Lens** is the primary interface between models, data, modules and (optionally) the Credo AI Governance Platform.
* **CredoModel / CredoData** are wrappers to bring models and data into the Lens Framework
* **Modules** are tools to perform assessment-related functions on models and/or data. A bias assessment is a type of module.
* **Assessments** are the connective tissue between CredoModels and CredoData and modules.

Usage of Lens boils down to creating the artifacts you want to assess (CredoModel and/or CredoData), articulating the
assessments you want run, how you want them to be run ("alignment") and running Lens. Most steps along this path
can be automated by Lens or fully customized by the user.

Lens
----
Lens is a single interface that allows easy assessment of your models and data.
Within the framework we have provided interfaces to well known responsible AI tools
as well as to Credo AI's own custom modules.

An "Alignment Spec" must be supplied to Lens. The spec articulates
how different assessments should be run at a high level. Essentially, 
it is a partial parameterization of the assessment.

The spec makes most sense as part of Credo AI's overall governance platform. In 
this context the Spec is the output of a multi-stakeholder articulation of
how the AI system should be assessed. Lens actually takes care of automatically
retrieving the Alignment Spec from the governance platform, connecting 
your technical team to compliance, product, and other stakeholders.

Of course, a single person can also defined the Spec. 
In this case, the Alignment Spec still serves as an initial decision
as to how assessments should be run, and summary of the assessments run.

Beyond the Alignment Spec, each module has other parameters that can be customized.
In addition, the set of modules are easily customized and extended. Lens strives to give
you sensible defaults without placing unnecessary restrictions. The `lens customization` tutorial
goes through these aspects.

Credo Model/Data
----------------
**Credo Models** are not "models" in the traditional sense - they are connector objects
that instantiate functions necessary for assessment. For instance, to evaluate
fairness usingn the "FairnessBase" assessment, the CredoModel must instantiate
a `prob_fun` or `pred_fun`. The nature of these functions can be quite general.

The simplest case is you setting `pred_fun` to the `predict` method of your model.
But your "model" may actually be an API call that you want to assess, in which case
the `pred_fun` may be an API call.

Some functions can be inferred from well-known frameworks like scikit-learn. This allows
the CredoModelt to be automatically set up, though further customization is possible.

**Credo Data** are simple objects that contain features (X), outputs (y) and (optionally) a sensitive feature.
Data assessments are run on these.

Credo AI Governance Platform
----------------------------
Assessment is important, but it's not the end all of Responsible AI development!
Creod AI's governance platform provides the other aspects needed for effective
AI governance including: support for multi-stakeholder alignment, policy packs
for different areas of responsible AI development and compliance needs,
and continuous translation of all evidence (including assessment result!) into
a risk perspective.

This platform is independent from using Lens for assessments. You can use *any*
method to assess your AI artifacts and upload the results to the Governance Platform
smoothly. Check out the `integration demo` to see how that is done.



