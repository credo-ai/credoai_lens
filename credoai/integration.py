"""Credo AI Governance App Integration Functionality"""

import base64
import io
import json
import mimetypes
import pprint
from collections import ChainMap, defaultdict
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
from absl import logging
from json_api_doc import deserialize

import credoai
from credoai.utils.common import (
    IntegrationError,
    ValidationError,
    dict_hash,
    humanize_label,
    wrap_list,
)
from credoai.utils.credo_api import CredoApi


META = {"source": "credoai_ml_library", "version": credoai.__version__}


class Record:
    def __init__(self, json_header, **labels):
        self.json_header = json_header
        # remove Nones from labels
        self.labels = {k: v for k, v in labels.items() if v != "NA"}
        self.creation_time = datetime.utcnow().isoformat()

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
    model_id : str, optional
        ID of model. Should match a model on Governance App
    dataset_id : str, optional
        ID of dataset. Should match a dataset on Governance App
    process : string, optional
        String reflecting the process used to create the metric. E.g.,
        name of a particular Lens assessment, or link to code.
    metadata : dict, optional
        Arbitrary structured data to append to metric
    labels : dict, optional
        Arbitrary keyword arguments to append to metric as metadata. These will be
        displayed in the governance app

    Example
    ---------
    metric = Metric('precision_score', 0.5)
    """

    def __init__(
        self,
        metric_type,
        value,
        subtype="base",
        model_id=None,
        dataset_id=None,
        process=None,
        metric_key=None,
        metadata=None,
        **labels
    ):
        super().__init__("metrics", **labels)
        self.metric_type = metric_type
        self.value = value
        self.subtype = subtype
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.process = process
        self.metadata = metadata or {} if metadata != "NA" else {}
        self.metadata.update({"model_id": self.model_id, "dataset_id": self.dataset_id})
        if metric_key:
            self.metric_key = metric_key
        else:
            self.metric_key = self._generate_config()

    def struct(self):
        return {
            "key": self.metric_key,
            "name": self.metric_type,
            "type": self.metric_type,
            "subtype": self.subtype,
            "value": self.value,
            "process": self.process,
            "labels": self.labels,
            "metadata": self.metadata,
            "value_updated_at": self.creation_time,
        }

    def _generate_config(self):
        ignored = ["value", "creation_time"]
        return dict_hash({k: v for k, v in self.__dict__.items() if k not in ignored})


class File(Record):
    def __init__(self, name, content, content_type, metric_keys=None, **labels):
        super().__init__("figures", **labels)
        self.name = name
        self.content = content
        self.content_type = content_type
        self.metric_keys = metric_keys
        self.content_type = None

    def struct(self):
        return {
            "name": self.name,
            "content": self.content,
            "content_type": self.content_type,
            "creation_time": self.creation_time,
            "metric_keys": self.metric_keys,
            "metadata": self.labels,
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
    metric_keys: list
        List of metric_keys to associate with figure (see lens_utils.get_metric_keys)
    labels : dict, optional
        Appended keyword arguments to append to metric as labels

    Example
    ---------
    f = plt.figure()
    figure = Figure('Figure 1', fig=f, description='A matplotlib figure')
    """

    def __init__(self, name, figure, description=None, metric_keys=None, **labels):
        super().__init__("figures", **labels)
        self.name = name
        self.description = description
        self.metric_keys = metric_keys
        self.figure_string = None
        self.content_type = None
        if type(figure) == matplotlib.figure.Figure:
            self._encode_matplotlib_figure(figure)
        else:
            self._encode_figure(figure)

    def _encode_figure(self, figure_file):
        with open(figure_file, "rb") as figure2string:
            self.figure_string = base64.b64encode(figure2string.read()).decode("ascii")
        self.content_type = mimetypes.guess_type(figure_file)[0]

    def _encode_matplotlib_figure(self, fig):
        pic_IObytes = io.BytesIO()
        fig.savefig(pic_IObytes, format="png", dpi=300, bbox_inches="tight")
        pic_IObytes.seek(0)
        self.figure_string = base64.b64encode(pic_IObytes.read()).decode("ascii")
        self.content_type = "image/png"

    def struct(self):
        return {
            "name": self.name,
            "description": self.description,
            "content_type": self.content_type,
            "file": self.figure_string,
            "creation_time": self.creation_time,
            "metric_keys": self.metric_keys,
            "metadata": {"type": "chart", **self.labels},
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


def record_metric(metric_type, value, **labels):
    """Convenience function to create a metric json object

    Parameters
    ----------
    metric_type : string
        short identifier for metric. credoai.utils.list_metrics provides
        a list of standard metric families.
    value : float
        metric value
    labels : dict, optional
        Arbitrary keyword arguments to append to metric as labels

    Returns
    -------
    Metric object
    """

    return Metric(metric_type, value, **labels)


def record_metrics(metric_df):
    """
    Function to create a list of metric json objects

    Metrics must be properly formatted in a pandas dataframe

    Parameters
    ------------
    metric_df : pd.DataFrame
        dataframe where the index is the metric name and the columns
        are passed to record_metric
    labels : dict, optional
        Arbitrary keyword arguments to append to metric as labels
    """
    records = []
    for metric, row in metric_df.iterrows():
        records.append(record_metric(metric_type=metric, **row))
    return MultiRecord(records)


def record_metrics_from_dict(metrics, **labels):
    """
    Function to create a list of metric json objects from dictionary

    All metrics will have the same labels using this function.
    To assign unique labels to each metric use
    `credoai.integration.record_metrics`

    Parameters
    ------------
    metrics : dict
        dictionary of metric_type : value pairs
    labels : dict, optional
        Arbitrary keyword arguments to append to metric as labels
    """
    if len(metrics) == 0:
        raise ValidationError("Empty dictionary of metrics provided")
    metric_df = pd.Series(metrics, name="value").to_frame()
    metric_df = metric_df.assign(**labels)
    return record_metrics(metric_df)


def prepare_assessment_payload(
    assessment_results, reporter_assets=None, assessed_at=None
):
    """Export assessment json to file or credo

    Parameters
    ----------
    assessment_results : dict or list
        dictionary of metrics to pass to record_metrics_from _dict or
        list of prepared_results from credo_assessments. See lens.export for example
    reporter_assets : list, optional
            list of assets from a CredoReporter, by default None
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
        assessment_records = (
            MultiRecord(assessment_records).struct() if assessment_records else {}
        )
    if reporter_assets:
        chart_assets = [asset for asset in reporter_assets if "figure" in asset]
        file_assets = [asset for asset in reporter_assets if "content" in asset]
        chart_records = [Figure(**assets) for assets in chart_assets]
        chart_records = MultiRecord(chart_records).struct() if chart_records else []
        file_records = [File(**assets) for assets in file_assets]
        file_records = MultiRecord(file_records).struct() if file_records else []
    else:
        chart_records = []
        file_records = []

    payload = {
        "assessed_at": assessed_at or datetime.utcnow().isoformat(),
        "metrics": assessment_records,
        "charts": chart_records,
        "files": file_records,
        "$type": "string",
    }
    return payload


def process_assessment_spec(spec_destination, api: CredoApi):
    """Get assessment spec from Credo's Governance App or file

    At least one of the credo_url or spec_path must be provided! If both
    are provided, the spec_path takes precedence.

    The assessment spec includes all information needed to assess a model and integrate
    with the Credo AI Governance Platform. This includes the necessary IDs, as well as
    the assessment plan

    Parameters
    ----------
    spec_destination: str
        Where to find the assessment spec. Two possibilities. Either:
        * end point to retrieve assessment spec from credo AI's governance platform
        * The file location for the assessment spec json downloaded from
        the assessment requirements of an Use Case on Credo AI's
        Governance App

    Returns
    -------
    dict
        The assessment spec, with artifacts ids and assessment plan
    """
    spec = {}
    try:
        spec = api.get_assessment_spec(spec_destination)
    except:
        spec = deserialize(json.load(open(spec_destination)))

    # reformat assessment_spec
    metric_dict = defaultdict(dict)
    metrics = spec["assessment_plan"]["metrics"]
    assessment_plan = defaultdict(list)
    for metric in metrics:
        bounds = (metric["lower_threshold"], metric["upper_threshold"])
        assessment_plan[metric["risk_issue"]].append(
            {"type": metric["metric_type"], "bounds": bounds}
        )
    spec["assessment_plan"] = assessment_plan
    return spec
