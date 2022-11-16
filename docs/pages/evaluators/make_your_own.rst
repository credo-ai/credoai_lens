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


*******************
Abstract class info
*******************

The full reference for the `Evaluator` class can be found :ref:`on this page<credoai.evaluators.evaluator.Evaluator>`.

:meth:`credoai.evaluators.Evaluator.results`

****************
Evaluator schema
****************


*********************
Docstring style guide
*********************


********************
Summary of the steps
********************