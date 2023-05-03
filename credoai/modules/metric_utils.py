import textwrap
from collections import defaultdict

from pandas import DataFrame

from credoai.modules.metrics import ALL_METRICS, METRIC_CATEGORIES


def list_metrics(verbose=True):
    metrics = defaultdict(set)
    for metric in ALL_METRICS:
        if metric.metric_category in METRIC_CATEGORIES:
            metrics[metric.metric_category] |= metric.equivalent_names
    if verbose:
        for key, val in metrics.items():
            metric_str = textwrap.fill(
                ", ".join(sorted(list(val))),
                width=80,
                initial_indent="\t",
                subsequent_indent="\t",
            )
            print(key)
            print(metric_str)
            print("")
    return metrics


def table_metrics():
    output = DataFrame(
        [
            [
                metric.name,
                metric.metric_category,
                list(metric.equivalent_names),
                metric.get_fun_doc,
            ]
            for metric in ALL_METRICS
        ],
        columns=["metric_name", "metric_category", "synonyms", "doc"],
    )

    def remove(list1, str1):
        list1 = [x for x in list1 if x != str1]
        return list1

    output["synonyms"] = output.apply(
        lambda row: remove(row.synonyms, row.metric_name), axis=1
    )
    output["synonyms"] = output.synonyms.apply(lambda x: ", ".join(x))
    return output
