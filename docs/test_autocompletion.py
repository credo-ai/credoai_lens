from credoai.modules.metric_utils import table_metrics


def create_title(title: str):
    output = f"\n{title.capitalize()}\n=======\n"
    return output


if __name__ == "__main__":
    a = table_metrics()[["metric_name", "metric_category", "synonyms"]].iloc[:2]
    a["sphinx"] = [
        "\n\t  - ".join(i)
        for i in zip(a["metric_name"], a["metric_category"], a.synonyms)
    ]
    a["sphinx"] = "\t* - " + a.sphinx

    with open("./metrics_v2.rst", "a") as text_file:
        for x in a.sphinx:
            text_file.write(x + "\n")
