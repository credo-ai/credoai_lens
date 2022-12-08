
Privacy (Experimental)
======================


Privacy module for Credo AI (Experimental)

This module takes  provides functionality to perform privacy assessment.

The main library leveraged for the purpose is the
`adversarial robustness toolbox <https://adversarial-robustness-toolbox.readthedocs.io/en/latest/>`_.
The types of attacks used by this evaluator are the following (click on the links for more info):

* `Attribute Inference Baseline`_: Trains a neural network to learn the attacked feature from the other features.
* `Attribute Inference BlackBox`_: Trains a neural network to learn the attacked feature from the other features and
  the model's prediction.
* `Membership Inference BlackBox`_: Trains a neural network to assess if some records were used for the model training.
* `Membership Inference BlackBox Rule Based`_: Use a simple rule based approach to assess if some records
  were used for the model training.

Parameters
----------
attack_feature : Union[str, int, None], optional
    Either the name or the column number of the feature to be attacked. If the column
    number is provided, the following parameter `attack_feature_name` needs to be provided.
    Default is None, in this case no attack feature is performed.
attack_feature_name : Optional[str], optional
    The name of the feature to be attacked, this is to be provided only in the case `attack_feature` is
    an integer. This allows for data like numpy.matrix that do not possess column names can be passed
    as datasets. By default None.
attack_train_ratio : float, optional
    Internally the train/test dataset are further split in order to train the models performing the
    attacks. This indicates the split ratio, by default 0.50

.. _Attribute Inference Baseline: https://adversarial-robustness-toolbox.readthedocs.
   io/en/latest/modules/attacks/inference/attribute_inference.html#attribute-inference-baseline
.. _Attribute Inference BlackBox: https://adversarial-robustness-toolbox.readthedocs.
   io/en/latest/modules/attacks/inference/attribute_inference.html#attribute-inference-black-box
.. _Membership Inference BlackBox Rule Based: https://adversarial-robustness-toolbox.readthedocs.
   io/en/latest/modules/attacks/inference/membership_inference.html#membership-inference-black-box-rule-based
.. _Membership Inference BlackBox: https://adversarial-robustness-toolbox.readthedocs.
   io/en/latest/modules/attacks/inference/membership_inference.html#membership-inference-black-box
