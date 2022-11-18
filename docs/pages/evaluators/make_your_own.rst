########################
How to create Evaluators
########################

Evaluators are the core components of the Lens framework. Each evaluator performs a specific type
of assessment such as performance, fairness, privacy, etc... You can find the list of all evaluators
in the page :ref:`Library of Evaluators`.

The following documents goes through: 

1. :ref:`structure of the abstract class<core evaluator methods>`
2. :ref:`general methods organization within an evaluator object<evaluator schema>`
3. :ref:`docstring style guide<docstring style guide>`
4. :ref:`list of recommended steps to create an evaluator<summary of the steps>`

In order to understand the structure of evaluators, it is important to understand how they
are used within :class:`~.credoai.lens.lens.Lens`. A typical example of how Lens consumes evaluator
can be the following:

.. code-block::

    lens = Lens(model=credo_model, assessment_data=credo_data)
    lens.add(Performance(metrics = 'false positive rate'))

In this snippet of code (an extract of the notebook :ref:`quickstart`) , Lens is initiated with a model and assessment_data, 
and a :class:`~.credoai.evaluators.performance.Performance` evaluator is added to the lens pipeline.
The Performance evaluator is also initiated, in this case the parameter `metrics` is provided. 
The objects credo_data and credo_model are example of :mod:`~.credoai.artifacts`.

Evaluators consume artifacts, they act on models/data in order to perform assessments. Multiple evaluators can be
added to a single Lens run. Each of them is an independent object, the assessment performed by one evaluator do not have
any effect on the results of the others.

The life cycle of an evaluator is the following:

#. The evaluator gets initialized with any necessary (and/or optional) parameter required for the evaluation.
   In the snippet above the parameter `metrics` is necessary for the Performance evaluator to function.

#. The initialized evaluator is added to a Lens instance.

#. Based on the evaluator property :attr:`~.credoai.evaluators.evaluator.Evaluator.required_artifacts`, 
   Lens provides to the evaluator the artifacts necessary for its working. A Lens instance can be initialized
   with only three artifacts: a model, training data and assessment data. An evaluator can potentially require
   any combination of artifacts.

#. A validation step is performed, by the evaluator, to make sure that the artifacts are structured
   in the correct way in order for the assessment to take place correctly. If the validation fails,
   Lens communicates to the user the reason behind the failure, and no assessment from that specific evaluator takes place.

#. If the validation is successful, the evaluator performs the assessment.
   

The steps above are performed by specific methods shared by all evaluators. In the next section we will explore the way
these methods are built and organized.

**********************
Core evaluator methods
**********************

The full API reference for the ``Evaluator`` class can be found :ref:`on this page<credoai.evaluators.evaluator.Evaluator>`.
The abstract class defines the main methods necessary for the functioning of an evaluator
within the Lens framework.

Mirroring the steps listed in the previous section, the methods and properties the user will have to define
are the following:

#. :meth:`~.credoai.evaluators.evaluator.Evaluator.__init__` -> Standard class initialization, the particularity
   is that there is no need to specify any of the Lens artifacts (model, training data, assessment data). Lens
   handle the passing of the artifacts to the evaluator.

#. :attr:`~.credoai.evaluators.evaluator.Evaluator.required_artifacts` -> The strings contained in this set,
   establish which artifacts Lens will try to pass to the evaluator, these artifacts will be made available
   in the `self` object of the evaluator.
   The accepted string values are:

   * ``"model"``: This means the evaluator requires a model. Accessible as ``self.model``
   * ``"assessment_data"``: This represent any dataset used to perform assessment. Accessible as ``self.assessment_data``
   * ``"training_data"``: This represent a dataset used to perform a model training during assessment time.
     Accessible as ``self.assessment_data``
   * ``"data"``: This means that the evaluator can work on any generic dataset. If both training and assessment
     data are available, Lens will run the evaluator on each separately.
   * ``"sensitive_feature"``: this is a special value, it represents a dependance of the evaluator on sensitive
     features, whereby that is intended in the context of Responsible AI. In case multiple sensitive features
     are available, Lens will run the evaluator on each separately.

#. :meth:`~.credoai.evaluators.evaluator.Evaluator._validate_arguments` -> Any validation on the format and content
   of the required artifacts should be performed in this method. The module :mod:`~.credoai.evaluators.utils.validation`
   contains several pre-made utility function that can aid the user in creating their validity checks.

#. :meth:`~.credoai.evaluators.evaluator.Evaluator._setup` -> This method is supposed to contain any extra step necessary
   to complete the initialization. This is introduced because the required artifacts are made available at a later
   time compared to when the evaluator class is initialized.

.. important::
   
   The methods ``_validate_arguments()`` and ``_setup()``, together with the passing of the artifacts are handled
   programmatically by Lens. The user must not explicitly call them from withing the evaluator. For the interested
   reader, this part of the automation is handled by Lens via the function :meth:`~.credoai.evaluators.evaluator.Evaluator.__call__`.

5. :meth:`~.credoai.evaluators.evaluator.Evaluator.evaluate()` -> This is the method that effectively runs all
   the evaluating procedure. The user is free to structure the running of the evaluation as preferred, there
   are no restriction on number of methods, however the method evaluate needs to be used to **run** the whole
   procedure. This is the method that Lens references internally.

   The ``evaluate()`` method populates the property :attr:`~.credoai.evaluators.evaluator.Evaluator.results`, this property
   can only accept list of `evidence containers <https://github.com/credo-ai/credoai_connect/blob/develop/connect/evidence/containers.py>`_.

****************
Evaluator schema
****************

This section deals with best practices in the organization of an evaluator class. This is a list
of principles that aim to make the structure of evaluators consistent, easy to interpret and debug.

#. Evaluator class naming is in CamelCase, consistent with Python best practices
#. Immediately after the class name a docstring describing the purpose of the evaluator and any
   parameter necessary at init time is inserted. For more info on the docstring structure, please
   refer to the next section.
#. 

*********************
Docstring style guide
*********************


********************
Summary of the steps
********************