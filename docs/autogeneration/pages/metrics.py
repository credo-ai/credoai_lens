from pandas import read_json
from credoai.modules.metric_utils import table_metrics
from docs.autogeneration.formatter import (
    convert_df_to_table,
    create_table,
    create_title,
    create_page_area,
)

INTRO = """Metrics new auto version
========================

Lens supports many metrics out-of-the-box. 
The following gives a comprehensive list, which you can also generate in your python environment:

Below we provide details for a selection of these supported metrics. 

Custom metrics are supported by using the `Metric` class, which can be used to wrap any assessment function.
"""


def create_metric_recap(row):
    parts = [create_title(row.metric_name, "section", False), row.description]
    if row.url:
        parts += [f"\n**Source**: `click here <{row.url}>`__"]
    if row.synonyms:
        parts += [f"\n**Other known names**: {row.synonyms}"]
    info = create_page_area(parts)
    return info


if __name__ == "__main__":

    # Mix auto-generated with manual metrics info
    df = table_metrics()
    manual_info = read_json("./metrics_info_manual.json")
    df = df.merge(manual_info)
    df = df.loc[~df.metric_name.duplicated()]

    # Create table of metrics
    df["Metric Name"] = ":ref:`" + df.metric_name + "<" + df.metric_name + ">`"
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
    with open("./metrics.rst", "w") as text_file:
        text_file.write(page)
