Metrics new auto version
=======

Lens supports many metrics out-of-the-box. 
The following gives a comprehensive list, which you can also generate in your python environment:

.. exec_code::

   from credoai.modules.metric_utils import list_metrics
   list_metrics()

Below we provide details for a selection of these supported metrics. 

Custom metrics are supported by using the `Metric` class, which can be used to wrap any assessment function.


Testing
=======
This is all just a test