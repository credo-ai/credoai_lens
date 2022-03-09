Tutorials
=========

A diverse set of demonstration notebooks are included with Lens that helps you get started and become familiar with its many capabilities.

Introductory Tutorials
----------------------
Start here!

.. toctree::
   :maxdepth: 1

   notebooks/quickstart
   notebooks/governance_integration
   notebooks/lens_customization

**Quickstart Demo**

Our journey starts here. We will assess a payment default prediction model for gender fairness 
using Lens, in 5 minutes.

**Connecting with the Credo AI Governance Platform Demo**

Lens is primarily a framework for comprehensive assessment of AI models. However, it is also the 
primary way to integrate assessment analysis with 
`Credo AI’s Governance Platform <https://www.credo.ai/product>`__. 
This notebook walks you through the steps to set up the connection.

**Customizing Lens Demo**

Lens strives to give you sensible defaults and automatically do the proper assessments. However, there are times where you want to customize a functionality. This notebook shows you how.

Lens Demos
----------
Demos of other use cases and assessments using the Lens framework.

.. toctree::
   :maxdepth: 1

   notebooks/lens_demos/binaryclassification
   notebooks/lens_demos/dataset_assessment
   notebooks/lens_demos/nlp_generator

**Binary Classification Demo**

Lens can help you assess their performance and fairness of binary classification models. 
In this notebook, this capability is demonstrated on a scenario where binary classification is 
used to predict the likelihood that an applicant will default on a credit-card loan.

**Data Assessment Demo**

Biases in data could result in biased algorithmic outcomes. Lens can assess a structured 
dataset for biases. This notebook demonstrates this capability on an income dataset.

**Language Generation Demo**

Lens assess language generation models like GPT3 for bias, toxicity, etc. See how in this notebook.


Module Demos
------------
Demos of using the modules directly. We **do not recommend** this method of interacting
with modules. However, it can be occasionally be useful if you know what you are doing!

.. toctree::
   :maxdepth: 1

   notebooks/module_demos/fainress_binaryclassification
   notebooks/module_demos/fairness_nlp
   notebooks/module_demos/nlp_generator_demo

**Binary Classification Demo**

Same as the Lens demo, but using the FairnessBase module directly.

**Word Embedding Models Bias Assessment Demo**

Word embeddings models generate a real-valued vector representation of text data and are 
mainstream in many AI systems that involve natural language data. However, they have been 
indicated to exhibit a range of human-like social biases. Lens is able to assess them for such 
biases. This capability is demonstrated in this demo.

**Language Generation Models Assessment Demo**

Language generation models generate meaningful text when prompted with a sequence of words as 
context and empower many modern applications, such as chatbots. This Lens module assesses a 
generation model for a text attribute (toxicity, profanity, etc.) and disparate impact. 
It has many prompts datasets and assessment models built in, but is also highly customizable. 
We will see how it works in action in this notebook through applying it to the popular 
GPT generation models.