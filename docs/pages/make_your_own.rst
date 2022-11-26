########################
How to create Evaluators
########################

Evaluators are the core components of the Lens framework. Each evaluator performs a specific type
of assessment such as performance, fairness, privacy, etc... You can find the list of all evaluators
in the page :ref:`Library of Evaluators`.

The following documents goes through: 

1. :ref:`Core methods for the class<core evaluator methods>`
2. :ref:`Evaluator class organization<evaluator schema>`
3. :ref:`Docstring style guide<docstring style guide>`
4. :ref:`TLTR: A condensed summary<summary of the steps>`

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
     data are available, Lens will run the evaluator on each separately. Accessible as ``self.data``.
   * ``"sensitive_feature"``: this is a special value, it represents a dependance of the evaluator on sensitive
     features, whereby that is intended in the context of Responsible AI. In case multiple sensitive features
     are available, Lens will run the evaluator on each separately. Accessible as ``self.sensitive_feature``

#. :meth:`~.credoai.evaluators.evaluator.Evaluator._validate_arguments` -> Any validation on the format and content
   of the required artifacts will be performed in this method. The module :mod:`~.credoai.evaluators.utils.validation`
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

In general we strive to follow `Python PEP8 <https://peps.python.org/pep-0008/>`_ guidelines.
Specifically to evaluator, these are the main directives:

#. All evaluators inherit the class :class:`~.credoai.evaluators.evaluator.Evaluator`.
#. Evaluator class naming is in CamelCase, consistent with Python best practices for classes.
#. Immediately after the class name a docstring describing the purpose of the evaluator and any
   parameter necessary at ``init`` time is inserted. For more info on the docstring structure, please
   refer to the next section.
#. Immediately after the docstring, these methods/property (enforced by the abstract class) follow in this order:

    #. ``__init__``: The last line of this method is ``super().__init__()``. The invocation of the abstract
       class init method is necessary to initialize some class properties.
    #. ``required_artifacts``: this is defined as a property of the class, outside of init
    #. ``_validate_arguments``
    #. ``_setup``
    #. ``evaluate``

#. The ``evaluate`` method is meant to be as simple as possible. Ideally the majority of the complexity is organized
   in a suitable amount of private methods.
#. By the end of the ``evaluate`` method, the property ``self.results`` needs to be populated with the results
   of the assessment.
#. Any other method created by the user to structure the code can come after evaluate. The only other recommendation
   is for static methods to be put after any non-static method.

*********************
Docstring style guide
*********************

This is a general style guide for any method docstring. In particular, the evaluator class docstring will be used to
create an evaluator specific page in :ref:`evaluators`, so following the guidelines will ensure that the page
will be displayed correctly.

The following settings are generally applied to any docstring. Modern IDE generally allow to configure
how docstrings will be populated. 

- Format: **numpy**
- Quote style: *"""* (3 x double quotes)
- Start on new line: True -> This forces the docstring to not start in line with the first
  3 double quotes. This setting is necessary for the docs page to be visualized correctly.

The default format for the text content within the docstring follows the `sphinx restructured text <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ 
conventions. Below an example of what a typical docstring could look like:

.. code-block::

   def MyEvaluator:
      """
      Evaluator purpose, no more than one line.

      Notice how the first line above starts below the triple quotes.

      Custom section
      --------------
      Any form of custom section is supported, and will
      be formatted according to rst rules. These sections can be used to further
      break down a lengthy description. NOtice the header level is defined by a rows
      of "-" the same length as the section title.

      Any section can contain numbered/bullet lists. There needs to be an empty line
      between the text and the start of the list. No indentation is required to start
      the list.

      1. Numbered item
      2. Another numbered item

      * Bullet point
        To extend to multiline simply align to the first letter
        * Sub bullet

      Parameters
      ----------
      param 1: type
         Description
      
      Examples
      --------

      This is expected to be the last section.

      Code syntax uses doctest conventions:
      1. Prefix lines with >>>, multiline code uses ... for second ine onward
      2. Line after a code line is interpreted as expected output
      3. Any output produced by the code needs to be matched for the test to succeed.
      See strategy below to bypass matching a specific output

      >>> a = 2
      >>> def my_func(): # multiline example
      ...   pass
      >>> print('123') # This outputs needs to be matched
      123

      To skip having to match a specific output:

      >>> import doctest
      >>> doctest.ELLIPSIS_MARKER = '-etc-'
      >>> # the next line produces output we will ignore
      >>> print(42) # doctest: +ELLIPSIS
      -etc-


      **WARNING!!!** Code prefixed with >>> will be tested during package testing, leveraging doctests
      capabilities.

      Pseudo code can be inserted using indentation, this will not be tested:

         my_pseudo_code = something_generic
      
      """

.. warning::

   It is necessary for any code prefixedto be syntactically correct and to conform to `doctests <https://docs.pytest.org/en/7.1.x/how-to/doctest.html>`_.
   You can find an example of a complex code section in :ref:`this docstring <Identity verification>`.

********************
Summary of the steps
********************
Here's a very practical, and condensed, approach to making an evaluator:

* Create a class that inherits from :class:`~.credoai.evaluators.evaluator.Evaluator`.
* Create the ``__init__``, remember that Lens artifacts are not meant to be here.
* Define the ``required_artifacts`` for the evaluator. 
* Define ``validate_arguments`` and ``setup``. These tend to be updated as the understanding.
  of the evaluator scope and desired outcome increase.
* Break down the logic of the evaluation in a suitable amount of methods.
* Finalize creating the ``evaluate`` method, which runs the full logic and populates ``self.results`` with
  a list of *evidence containers*.

During building/testing phase you can run the evaluator outside of Lens, in order to make the artifacts
available to the evaluator you can use the ``__call__`` method. An example of what a test would look like
would be:

.. code::

    test = MyEvaluator(param1 = value1)
    # If MyEvaluator requires model and assessment data, call the evaluator
    # to pass teh artifacts. This mimics what happens internally in Lens.
    test(model = my_model_artifact, assessment_data = my_assessment_data)
    test.evaluate()
    # To check the results
    test.results

