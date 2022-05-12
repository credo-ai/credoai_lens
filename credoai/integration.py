"""Credo AI Governance App Integration Functionality"""

from absl import logging
from collections import ChainMap, defaultdict
from datetime import datetime
from credoai.utils.common import (humanize_label, wrap_list,
                                  IntegrationError,
                                  ValidationError, dict_hash)
from credoai.utils.credo_api_utils import (get_assessment_plan,
                                           post_assessment,
                                           register_dataset, 
                                           register_model,
                                           register_model_to_usecase,
                                           register_dataset_to_model,
                                           register_dataset_to_model_usecase)
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


class Record:
    def __init__(self, json_header, **metadata):
        self.json_header = json_header
        # remove Nones from metadata
        self.metadata = {k:v for k, v in metadata.items() if v!='NA'}
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
    subtype : string, optional
        subtype of metric. Defaults to base
    dataset_id : str, optional
        ID of dataset. Should match a dataset on Governance App
    process : string, optional
        String reflecting the process used to create the metric. E.g.,
        name of a particular Lens assessment, or link to code.
    metadata : dict, optional
        Arbitrary keyword arguments to append to metric as metadata. These will be
        displayed in the governance app

    Example
    ---------
    metric = Metric('precision_score', 0.5)
    """

    def __init__(self,
                 metric_type,
                 value,
                 subtype="base",
                 dataset_id=None,
                 process=None,
                 **metadata):
        super().__init__('metrics', **metadata)
        self.metric_type = metric_type
        self.value = value
        self.subtype = subtype
        self.dataset_id = dataset_id
        self.process = process
        self.config_hash = self._generate_config()

    def struct(self):
        return {
            'key': self.config_hash,
            'name': self.metric_type,
            'type': self.metric_type,
            'subtype': self.subtype,
            'value': self.value,
            'dataset_id': self.dataset_id,
            'process': self.process,
            'labels': self.metadata,
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
            raise ValidationError("Individual records must all be of the same type")
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
    if len(metrics) == 0:
        raise ValidationError("Empty dictionary of metrics provided")
    metric_df = pd.Series(metrics, name='value').to_frame()
    metric_df = metric_df.assign(**metadata)
    return record_metrics(metric_df)

def prepare_assessment_payload(assessment_results, report=None, assessed_at=None):
    """Export assessment json to file or credo

    Parameters
    ----------
    assessment_results : dict or list
        dictionary of metrics to pass to record_metrics_from _dict or
        list of prepared_results from credo_assessments. See lens.export for example
    report : credo.reporting.NotebookReport, optional
        report to optionally include with assessments, by default None
    assessed_at : str, optional
        date when assessments were created, by default None
    for_app : bool
        Set to True if intending to send to Governance App via api
    """    
    # prepare assessments
    if isinstance(assessment_results, dict):
        assessment_records = record_metrics_from_dict(assessment_results).struct()
    else:
        assessment_records = [record_metrics(r) for r in assessment_results]
        assessment_records = MultiRecord(assessment_records).struct() if assessment_records else {}

    # set up report
    default_html = '<html><body><h3 style="text-align:center">No Report Included With Assessment</h1></body></html>'
    report_payload = {'content': default_html,
                      'content_type': "text/html"}
    if report:
        report_payload['content'] = report.to_html()
    
    payload = {"assessed_at": assessed_at or datetime.now().isoformat(),
               "metrics": assessment_records,
               "charts": [],
               "report": report_payload,
               "$type": 'string'}
    return payload


def get_assessment_spec(use_case_id=None, model_id=None, spec_path=None):
    """Get aligned metrics from Credo's Governance App or file

    At least one of the use_case_id or spec_path must be provided! If both
    are provided, the spec_path takes precedence.

    Parameters
    ----------
    use_case_id : str, optional
        ID of Use Case on Credo AI Governance app, by default None
    model_id : str, optional
        ID of model on Credo AI Governance app, by default None
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
    if spec_path:
        spec = json.load(open(spec_path))
    elif use_case_id:
        try:
            spec = get_assessment_plan(use_case_id, model_id)
        except IntegrationError:
            logging.warning(f"No spec found for model ({model_id}) under model use case ({use_case_id})")
            return spec
    metric_dict = defaultdict(dict)
    metrics = spec['metrics']
    risk_spec = defaultdict(list)
    for metric in metrics:
        bounds = (metric['lower_threshold'], metric['upper_threshold'])
        risk_spec[metric['risk_issue']].append({'type': metric['type'], 'bounds': bounds})
    return risk_spec
