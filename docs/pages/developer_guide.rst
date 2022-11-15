
Developer Guide
===============

This document will go into more detail about the different components of Lens so that you can
extend it to fit your needs. 

Lens reuses a design pattern whereby abstract classes are defined which articulate base 
functionality that are then extended by specific classes. For instance, all evaluators 
inherit from the ``Evaluator`` abstract class, models from the ``Model`` class, etc. 
Thus understanding the different abstract classes goes a long way to understanding the 
architecture of Lens. You can then build your own new model or evaluator by inheriting from
the base classes.

Abstract classes
----------------

Model
   The ``Model`` abstract class is used to wrap trained models. Classes inheriting
   ``Model`` will define specific functionality associated with a particularly kind of model type.
   For example, ``ClassificationModel`` defines the functionality expected of a classification model -
   specifically, that it has a ``predict`` function and possibly a ``predict_proba`` function.
   Model and Data objects are referred to as AI Artifacts (or just artifacts) in parts of Lens
   documentation.

Data
   The ``Data`` abstract class is used to wrap datasets. Classes inheriting ``Data`` will specify
   processing steps and validation to ensure that a a particular kind of dataset is standardized
   for use by downstream evaluators. ``TabularData`` is such a class and has certain requirements
   for the form of data passed to it, and ensures that the data has a certain form after processing
   (pandas dataframes or series, in this case). Model and Data objects are referred to as 
   AI Artifacts (or just artifacts) in parts of Lens documentation.

Evaluator
   The ``Evaluator`` abstract class is the workhorse of the framework. Added to a Lens pipeline,
   any Evaluator takes in AI Artifacts and outputs ``EvidenceContainers``.
   Classes inheriting ``Evaluator`` will perform any kind of assessment or function needed to
   understand an AI Artifacts behavior or characteristics. There are clearly a wide range of possible
   evaluators! Everything from assessing performance metrics to levying adversarial attacks
   can be instantiated in an evaluator.

EvidenceContainers
   The ``EvidenceContainer`` abstract class is used to hold results of evaluators. An EvidenceContainer
   is an extension of a pandas dataframe using the `composition strategy recommended by pandas <https://pandas.pydata.org/docs/development/extending.html#subclassing-pandas-data-structures>`_.
   Essentially, they contain a dataframe, but include addition functionality including validation
   and the ability to transform the dataframe into ``Evidence`` using the ``to_evidence`` function.
   ``EvidenceContainers`` are primarily used to ensure that all evaluator results can work with the
   Credo AI Governance Platform, which specifically consumes ``Evidence``.
   An example class inheriting ``EvidenceContainer`` is ``MetricContainer``, which contains
   a dataframe where each row is a different metric.

Evidence
   The ``Evidence`` abstract class is used to define the structure of results that can
   be consumed by the Credo AI Governance Platform. ``Evidence`` is automatically created
   from the associated ``EvidenceContainer`` and so is rarely interacted with directly by 
   a developer. Examples of ``Evidence`` include ``MetricEvidence`` which structure metric
   information in a particular JSON format for the platform.

Other Classes
-------------

Lens
   ``Lens`` is the primary interface between models, data, and evaluators. Lens performs an orchestration
   function. It interacts with an (optional) Governance object, generates or ingests a user-specified
   pipeline of evaluators, runs those evaluators, and defined functions to access the results.

   The pipeline of evaluators is a list of instantiated ``Evaluators`` with optional IDs or metadata
   attached. Manually, it can be created all at once by defining a list of evaluators, or incrementally,
   by adding evaluators to a Lens object one at a time. It can also be created automatically, using
   a ``PipelineCreator`` under the hood, in certain circumstances.

PipelineCreator
   A ``PipelineCreator`` handles pipeline creation that isn't manual. Currently it supports creating
   piplines when a ``Governance`` object is passed connected to a policy pack on the Platform. 

Governance
   The ``Governance`` object serves as a connection point between the Credo AI Platform and Lens.
   Developers who don't work specifically with Credo AI should not need to change this.

Developing in Lens
------------------

Section in progress! For now, the main area to extend functionality is by creating
your own Evaluator. 
