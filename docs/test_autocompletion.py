from credoai.modules.metric_utils import table_metrics
from autogeneration.formatter import (
    convert_df_to_table,
    create_table,
    create_title,
    create_page,
)

INTRO = """Metrics new auto version
========================

Lens supports many metrics out-of-the-box. 
The following gives a comprehensive list, which you can also generate in your python environment:

Below we provide details for a selection of these supported metrics. 

Custom metrics are supported by using the `Metric` class, which can be used to wrap any assessment function.
"""

if __name__ == "__main__":

    # Create table of metrics
    df = table_metrics().iloc[:2]
    df["name_hyperlink"] = ":ref:`" + df.metric_name + "<" + df.metric_name + ">`"
    metrics_table = create_table(
        convert_df_to_table(df, ["name_hyperlink", "metric_category", "synonyms"]),
        "List of all metrics",
    )

    # Create list of metrics section
    metrics_info = "\n".join(
        [create_title(m_name, "section", True) for m_name in df.metric_name]
    )

    # Create final page
    page = create_page([INTRO, metrics_table, metrics_info])

    # Create the page
    with open("./metrics_v2.rst", "w") as text_file:
        text_file.write(page)
