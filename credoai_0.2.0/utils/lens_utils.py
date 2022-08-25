from credoai.utils.common import dict_hash


def add_metric_keys(prepared_results):
    """Adds metric keys to prepared results

    Metric keys are used to associated charts, html blobs, and other assets with
    specific metrics. They are a hash of most of the metric's attributes, except the value.
    So if a metric changes value, the key will stay the same.

    Metric keys should be defined after all pertinent information is appended to a metric.
    Lens normally handles key association, because it may add additional metadata to a metric
    beyond what the assessment creates (e.g., dataset name, model name, etc.)

    Parameters
    ----------
    prepared_results : DataFrame
        output of CredoAssessment.prepare_results()
    """
    if prepared_results is None:
        return
    ignored = ["value", "metadata"]
    keys = [
        dict_hash({k: v for k, v in metric_dict.items() if k not in ignored})
        for metric_dict in prepared_results.reset_index().to_dict("records")
    ]
    prepared_results["metric_key"] = keys
