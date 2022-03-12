"""Credo AI Governance App Integration Functionality"""

from collections import ChainMap, defaultdict
from datetime import datetime
from json_api_doc import serialize
from credoai.utils.common import (NumpyEncoder, wrap_list,
                                  ValidationError, dict_hash)
from credoai.utils.credo_api_utils import (get_technical_spec,
                                           post_assessment,
                                           register_dataset, register_model,
                                           register_project,
                                           register_model_to_use_case)
import base64
import credoai
import json
import io
import matplotlib
import mimetypes
import numpy as np
import pandas as pd
import pprint

META = {
    'source': 'credoai_ml_library',
    'version': credoai.__version__
}
RISK = "fairness"


class Record:
    def __init__(self, json_header, **metadata):
        self.json_header = json_header
        self.metadata = metadata
        self.creation_time = datetime.now().isoformat()

    def struct(self):
        pass

    def __str__(self):
        return pprint.pformat(self.struct())


class Metric(Record):
    """
    A metric record

    Record of a metric 

    Parameters
    ----------
    metric_type : string
        short identifier for metric. credoai.utils.list_metrics provides
        a list of standard metric families.
    value : float
        metric value
    name : string, optional
        Specific identifier for particular metric. Defaults to the metric_type
    process : string, optional
        String reflecting the process used to create the metric. E.g.,
        name of a particular Lens assessment, or link to code.
    metadata : dict, optional
        Arbitrary keyword arguments to append to metric as metadata

    Example
    ---------
    metric = Metric('precision_score', 0.5)
    """

    def __init__(self,
                 metric_type,
                 value,
                 name=None,
                 process=None,
                 **metadata):
        super().__init__('metrics', **metadata)
        self.metric_type = metric_type
        self.value = value
        self.name = name or ' '.join(self.metric_type.split('_')).title()
        self.process = process
        self.config_hash = self._generate_config()

    def struct(self):
        return {
            'key': self.config_hash,
            'type': self.metric_type,
            'name': self.name,
            'value': self.value,
            'process': self.process,
            'metadata': self.metadata,
            'value_updated_at': self.creation_time,
        }

    def _generate_config(self):
        ignored = ['value', 'creation_time']
        return dict_hash({k: v for k, v in self.__dict__.items()
                          if k not in ignored})


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

    def __init__(self, name, figure, description=None, **metadata):
        super().__init__('figures', **metadata)
        self.name = name
        self.description = description
        self.figure_string = None
        self.content_type = None
        if type(figure) == matplotlib.figure.Figure:
            self._encode_matplotlib_figure(figure)
        else:
            self._encode_figure(figure)
            

    def _encode_figure(self, figure_file):
        with open(figure_file, "rb") as figure2string:
            self.figure_string = base64.b64encode(
                figure2string.read()).decode('ascii')
        self.content_type = mimetypes.guess_type(figure_file)[0]

    def _encode_matplotlib_figure(self, fig):
        pic_IObytes = io.BytesIO()
        fig.savefig(pic_IObytes,  format='png')
        pic_IObytes.seek(0)
        self.figure_string = base64.b64encode(
            pic_IObytes.read()).decode('ascii')
        self.content_type = "image/png"

    def struct(self):
        return {'name': self.name,
                'description': self.description,
                'content_type': self.content_type,
                'file': self.figure_string,
                'creation_time': self.creation_time,
                'metadata': {'type': 'chart', **self.metadata}
               }


class MultiRecord(Record):
    """
    A Multi-record object

    Stores multiple records of the same type

    Example
    ---------
    metric = Metric('recall', 0.6)
    model_record = MutliRecord(metric)
    """

    def __init__(self, records):
        self.records = wrap_list(records)
        if len(set(type(r) for r in self.records)) != 1:
            raise ValidationError
        super().__init__(self.records[0].json_header)

    def struct(self):
        data = [m.struct() for m in self.records]
        if isinstance(self.records[0], MultiRecord):
            data = [item for sublist in data for item in sublist]
        return data


def record_metric(metric_type, value,  **metadata):
    """Convenience function to create a metric json object

    Parameters
    ----------
    metric_type : string
        short identifier for metric. credoai.utils.list_metrics provides
        a list of standard metric families.
    value : float
        metric value
    metadata : dict, optional
        Arbitrary keyword arguments to append to metric as metadata

    Returns
    -------
    Metric object
    """

    return Metric(metric_type,
                  value,
                  **metadata)


def record_metrics(metric_df):
    """
    Function to create a list of metric json objects

    Metrics must be properly formatted in a pandas dataframe

    Parameters
    ------------
    metric_df : pd.DataFrame
        dataframe where the index is the metric name and the columns
        are passed to record_metric
    metadata : dict, optional
        Arbitrary keyword arguments to append to metric as metadata
    """
    records = []
    for metric, row in metric_df.iterrows():
        records.append(record_metric(metric_type=metric, **row))
    return MultiRecord(records)


def record_metrics_from_dict(metrics, **metadata):
    """
    Function to create a list of metric json objects from dictionary

    All metrics will have the same metadata using this function. 
    To assign unique metadata to each metric use 
    `credoai.integration.record_metrics`

    Parameters
    ------------
    metrics : dict
        dictionary of metric_type : value pairs
    metadata : dict, optional
        Arbitrary keyword arguments to append to metric as metadata
    """
    metric_df = pd.Series(metrics, name='value').to_frame()
    metric_df = metric_df.assign(**metadata)
    return record_metrics(metric_df)

def prepare_assessment_payload(prepared_results, report=None, assessed_at=None):
    """Export assessment json to file or credo

    Parameters
    ----------
    prepared_results : list
        prepared of prepared_results from credo_assessments. See lens.export_assessments for example
    report : credo.reporting.NotebookReport, optional
        report to optionally include with assessments, by default None
    assessed_at : str, optional
        date when assessments were created, by default None
    """    
    # prepare assessments
    assessment_records = [record_metrics(r) for r in prepared_results]
    assessment_records = MultiRecord(assessment_records)

    # set up report
    default_html = '<html><body><h3 style="text-align:center">No Report Included With Assessment</h1></body></html>'
    report_payload = {'content': default_html,
                      'content_type': "text/html"}
    if report:
        report_payload['content'] = report.to_html()
    
    payload = {"assessed_at": assessed_at or datetime.now().isoformat(),
               "metrics": assessment_records.struct(),
               "charts": None,
               "report": report_payload,
               "type": RISK,
               "$type": 'string'}

    payload_json = json.dumps(serialize(data=payload), cls=NumpyEncoder)
    return payload_json


def get_assessment_spec(use_case_id=None, spec_path=None, version='latest'):
    """Get aligned metrics from Credo's Governance App or file

    At least one of the use_case_id or spec_path must be provided! If both
    are provided, the spec_path takes precedence.

    Parameters
    ----------
    use_case_id : string, optional
        Identifier for Use Case on Credo AI's Governance App
    spec_path : string, optional
        The file location for the technical spec json downloaded from
        the technical requirements of an Use Case on Credo AI's
        Governance App
    Returns
    -------
    dict
        The aligned metrics for each model contained in the Use Case.
        Format: {"Model": {"Metric1": (lower_bound, upper_bound), ...}}
    """
    spec = {}
    if use_case_id:
        spec = get_technical_spec(use_case_id, version=version)
    if spec_path:
        spec = json.load(open(spec_path))
    metric_dict = defaultdict(dict)
    metrics = spec['metrics']
    for metric in metrics:
        bounds = (metric['lower_threshold'], metric['upper_threshold'])
        metric_dict[metric['model_id']][metric['type']] = bounds
    return metric_dict
