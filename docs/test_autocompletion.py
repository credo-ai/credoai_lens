from credoai.modules.metric_utils import table_metrics
from autogeneration.formatter import convert_df_to_table, create_table


if __name__ == "__main__":

    df = table_metrics().iloc[:2]
    with open("./metrics_v2.rst", "a") as text_file:
        text_file.write(
            create_table(
                convert_df_to_table(df, ["metric_name", "metric_category", "synonyms"]),
                "Testing",
            ),
        )
