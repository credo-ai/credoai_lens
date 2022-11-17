########################
How to create Evaluators
########################

Evaluators are the core components of the Lens framework. Each evaluator performs a specific type
of assessment such as performance, fairness, privacy, etc... You can find the list of all evaluators
in the page :ref:`Library of Evaluators`.

The following documents goes through: 

1. :ref:`structure of the abstract class<abstract class info>`
2. :ref:`general methods organization within an evaluator object<evaluator schema>`
3. :ref:`docstring style guide<docstring style guide>`
4. :ref:`list of recommended steps to create an evaluator<summary of the steps>`

In order to understand the structure of evaluators, it is important to understand how they
are used within :class:`~.credoai.lens.lens.Lens`. A typical example of how Lens consumes evaluator
can be the following:

.. code-block::

    lens = Lens(model=credo_model, assessment_data=credo_data)
    lens.add(Performance(metrics = 'false positive rate'))

In this snippet of code, Lens is initiated with a model and assessment_data, and a :class:`~.credoai.evaluators.performance.Performance`
evaluator is added to the lens pipeline. The Performance evaluator is also initiated, in this case
the parameter `metrics` is provided. A fully functioning example of a Lens run can be found in :ref:`quickstart`.



*******************
Abstract class info
*******************

The full API reference for the `Evaluator` class can be found :ref:`on this page<credoai.evaluators.evaluator.Evaluator>`.
The abstract class defines the main methods necessary for the functioning of an evaluator
within the Lens framework.

The important concepts to remember when creating an evaluator are the following:
:attr:`~.credoai.evaluators.Evaluator.required_artifacts`

****************
Evaluator schema
****************


*********************
Docstring style guide
*********************


********************
Summary of the steps
********************