from pathlib import Path

from pandas import read_json

from credoai.modules.metric_utils import table_metrics
from docs.autogeneration.formatter import (
    convert_df_to_table,
    create_page_area,
    create_table,
    create_title,
)

INTRO = """Metrics new auto version
========================

Lens supports many metrics out-of-the-box. 
The following gives a comprehensive list, which you can also generate in your python environment:

Below we provide details for a selection of these supported metrics. 

Custom metrics are supported by using the `Metric` class, which can be used to wrap any assessment function.
"""


def create_metric_recap(row):
    description = row.description
    if not row.description and row.doc() is not None:
        description = row.doc()
    parts = [create_title(row.metric_name, "section", False), description]
    if row.url:
        parts += [f"\n**Source**: `click here <{row.url}>`__"]
    if row.synonyms:
        parts += [f"\n**Other known names**: {row.synonyms}"]
    info = create_page_area(parts)
    return info


def create_metrics_page():
    # Mix auto-generated with manual metrics info
    df = table_metrics()
    manual_info_path = Path(__file__).parent.resolve() / "metrics_info_manual.json"
    manual_info = read_json(manual_info_path)

    df = df.merge(manual_info, how="left")
    df = df.loc[~df.metric_name.duplicated()]
    df = df.fillna("")

    # Create table of metrics
    df["Metric Name"] = ":ref:`" + df.metric_name + "<" + df.metric_name + ">`"
    df = df.sort_values(by=["Metric Name"])

    metrics_table = create_table(
        convert_df_to_table(df, ["Metric Name", "rai_dimension", "synonyms"]),
        title="List of all metrics",
        header=True,
    )

    # Create list of metrics section
    df["metric_info"] = df.apply(create_metric_recap, axis=1)
    metric_info = "\n".join(list(df.metric_info))

    # Create final page
    page = create_page_area([INTRO, metrics_table, metric_info])

    # Create the page
    with open("./pages/metrics.rst", "w") as text_file:
        text_file.write(page)


if __name__ == "__main__":
    create_metrics_page()
