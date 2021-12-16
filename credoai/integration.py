from collections import ChainMap
from datetime import datetime
from json_api_doc import serialize
from credoai.utils.common import NumpyEncoder, wrap_list
  
import base64
import credoai
import json
import io 
import matplotlib
import numpy as np
import pprint

META = {
  'source': 'credoai_ml_library',
  'version': credoai.__version__
}

    
class Record:
    def __init__(self, **metadata):
        self.metadata = metadata
        self.creation_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        
    def _struct(self):
        pass
    
    def _credoify(self):
        return json.dumps(serialize(data=self._struct()), cls=NumpyEncoder)
    
    def jsonify(self, filename=None):
        data = self._struct()
        if '$type' in data:
            del data['$type']
        return json.dumps({'data': data}, cls=NumpyEncoder)
    
    def __str__(self):
        return pprint.pformat(self._struct())
    
class Metric(Record):
    """
    A metric record

    Records a metric value. Added to a metric table for the relevant
    control.

    Parameters
    ----------
    name : string
        short identifier for metric
    value : float
        metric value
    model_label : string
        label of model version. Could indicate a specific model class,
        e.g., logistic_regression_1.0, or simply a version if the type
        of model is obvious (e.g., 1.2)
    dataset_label : string
        label of dataset
    user_id : str, optional
        identifier of metric creator
    metadata : dict, optional
        Appended keyword arguments to append to metric as metadata

    Example
    ---------
    metric = Metric('precision', 0.5)
    """
    def __init__(self,
            name,
            metric_family,
            value,
            model_label,
            dataset_label,
            **metadata):
        super().__init__(**metadata)
        self.name = name
        self.metric_family = metric_family
        self.value = value
        self.model_label = model_label
        self.dataset_label = dataset_label
    
    def _struct(self):
        return {
            'family': self.metric_family,
            'name': self.name,
            'value': self.value,
            'model_version': self.model_label,
            'dataset': self.dataset_label,
            'metadata': {'creation_time': self.creation_time,
                        **self.metadata},
            '$type': 'model_metrics'
        }
    
class Figure(Record):
    """
    A figure record

    Records a figure (either an image or a matplotlib figure).
    Attached as a figure to the associated control

    Parameters
    ----------
    name : string
        title of figure
    figure: str OR matplotlib figure
        path to image file OR matplotlib figure
    description: str, optional
        longer string describing the figure
    metadata : dict, optional
        Appended keyword arguments to append to metric as metadata

    Example
    ---------
    f = plt.figure()
    figure = Figure('Figure 1', fig=f, description='A matplotlib figure')
    """
    def __init__(self, name, figure, description=None, metadata=None):
        super().__init__(metadata)
        self.name = name
        self.description = description
        self.metadata = {} if metadata is None else metadata
        if type(figure) == matplotlib.figure.Figure:
            self._encode_matplotlib_figure(figure)
        else:
            self._encode_image_file(figure)
            

    def _encode_image_file(self, image_file):
        with open(image_file, "rb") as image2string:
            self.image_string = base64.b64encode(image2string.read()).decode('ascii')
            
    def _encode_matplotlib_figure(self, fig):
        pic_IObytes = io.BytesIO()
        fig.savefig(pic_IObytes,  format='png')
        pic_IObytes.seek(0)
        self.image_string = base64.b64encode(pic_IObytes.read()).decode('ascii')
        
    def _struct(self):
        return {'name': self.name,
                'description': self.description,
                'file': self.image_string,
                'creation_time': self.creation_time,
                'metadata': {'type': 'chart', **self.metadata},
                '$type': 'model_assets'}
    
class ModelRecord(Record):
    """
    A model record

    Stores metrics, figures, inputs, and outputs associated with a model

    Example
    ---------
    metric = Metric('recall', 0.6)
    features = [Input('categorical feature 1', None, 'The first categorical feature'),
                Input('categorical feature 2', None, 'The second categorical feature')]
    model_record = ModelRecord(metric, io_reocrds=features)
    """
    
    def __init__(self, metric_records=None, figure_records=None, io_records=None):
        super().__init__()
        self.metrics = []
        self.figures = []
        if metric_records:
            self.record_metrics(metric_records)
        if figure_records:
            self.record_figures(figure_records)
    
    def record_metrics(self, metric_records):
        """Records a metric record, or list of metric records"""
        self.metrics = wrap_list(metric_records)
        
    def record_figures(self, figure_records):
        """Records a figure record, or list of figure records"""
        self.figures = wrap_list(figure_records)
        
    def _struct(self):
        # metrics
        data = [m._struct() for m in self.metrics] 
        
        # other attributes that aren't incorporated yet
        extras = [f._struct() for f in self.figures] # figures
        return data
    
def create_records(record_class, record_args):
    """
    Convenience function to create a list of records
    
    Parameters
    ----------
    record_class : a Record class
    record_args : dict
        Of the form {kwarg1: list1, kwarg2: list2, ...}
        Each kwarg corresponds to an argument for the record_class.
        Each list should be the same length. Use None if an argument
        doesn't apply to a metric
        
    Returns
    ---------
    record_list : list
        list of record objects
        
    Example
    ---------
    record_args = {'name': ['metric1', 'metric2'], 
                   'value': [0.5, 0.6],
                   'value_range': [None, (0, 1)]}
    record_list = create_records(Metric, record_args)
    """
    n_records = len(list(record_args.values())[0])
    all_records = []
    for i in range(n_records):
        record_kwargs = {k: v[i] for k, v in record_args.items()}
        all_records.append(record_class(**record_kwargs))
    return all_records  

def create_metric_records(metric_args): 
    """
    Convenience function to create a list of metric records
    
    Parameters
    ----------
    record_args : dict
        Of the form {kwarg1: list1, kwarg2: list2, ...}
        Each kwarg corresponds to an argument for a Metric.
        Each list should be the same length. Use None if an argument
        doesn't apply to a metric
        
    Returns
    ---------
    record_list : list
        list of record objects
        
    Example
    ---------
    metric_args = {'name': ['metric1', 'metric2'], 
                   'value': [0.5, 0.6],
                   'description': ["A description of metric1", ""]}
    metric_list = create_metric_records(metric_args)
    """
    
    return create_records(Metric, metric_args)


def record_metric(name, metric_family, value, model_label, dataset_label, **metadata):
    """Convenience function to create a metric json object

    Parameters
    ----------
    name : string
        short identifier for metric
    value : float
        metric value
    model_label : string
        label of model version. Could indicate a specific model class,
        e.g., logistic_regression_1.0, or simply a version if the type
        of model is obvious (e.g., 1.2)
    dataset_label : string
        label of dataset
    user_id : str, optional
        identifier of metric creator

    Returns
    -------
    Metric object
    """    

    return Metric(name, metric_family, 
                  value, model_label, \
                  dataset_label,  **metadata)

def record_metrics(metrics):
    """
    Convenience function to create a list of metric json objects
    
    ** ATTENTION **
    Note this function cannot supply different
    models, datasets, user_ids or metadata for different metrics.
    For those use cases, use create_metric_records
    
    Parameters
    ------------
    metrics : pd.DataFrame
        dataframe where the index is the metric name and the columns
        are passed to record_metric
    """
    records = []
    for metric, row in metrics.iterrows():
        records.append(record_metric(name=metric, **row))
    return records

def export_record(record, filename):
    """Exports record as json object to filename

    Parameters
    ----------
    record : credo.integration.Record
        A Record object
    filename : str
        file to write Record json object
    """    
    json_out = record.jsonify()
    with open(filename, 'w') as f:
        f.write(json_out)