..
   Note: Items in this toctree form the top-level navigation. See `api.rst` for the `autosummary` directive, and for why `api.rst` isn't called directly.

.. toctree::
   :hidden:

   Home page <self>
   Jupyter tutorials <tutorials>
   API reference <_autosummary/credoai>

CredoAI | Lens
===============

CredoAI Lens is CredoAI's AI Assessment Framework. It supports assessment
of AI systems, with a focus on responsible AI assessment. It also
streamlines integration with the Credo AI governance platform.

Overview
--------
CredoAI Lens is made of a few components.

* Lens: the primary interface between models, data, modules and (optionally) the Credo AI Governance Platform.
* CredoModel / CredoData: wrappers to bring models and data into the Lens Framework
* Modules: arbitrary tools to perform assessment-related functions on models and/or data. A bias assessment is a type of module.
* Assessments: "assessments" are the connective tissue between CredoModels and CredoData and modules.


Lens
----
Lens is a single interface that allows easy assessment of your models and data.
Within the framework we have provided interfaces to well known responsible AI tools
as well as to CredoAI's own custom tooling. The set of modules are easily customized and
extended.

