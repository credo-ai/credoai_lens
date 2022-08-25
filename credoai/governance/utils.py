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
