"""
Sphinx mkd formatting functinality.

This library contains all the utility functions necessacry to format
content in a suitable way for a sphinx rst file.
"""
from typing import List, Literal, Optional

from pandas import DataFrame, Series, concat


def create_title(
    title: str, level: Literal["title", "section"] = "title", hyperlink: bool = False
):
    if level == "title":
        sep = "="
    elif level == "section":
        sep = "-"
    else:
        raise ValueError("Unknown level.")

    ttl_length = len(title)
    title_string = f"\n{title.capitalize()}\n{sep*ttl_length}\n"
    if hyperlink:
        title_string = f"\n.. _{title}:\n" + title_string

    return title_string


def create_table(instructions: str, title: Optional[str] = None, header: bool = False):
    table_title = ".. list-table::"
    table_header = "\t:header-rows: 1"
    if title:
        table_title += f" {title}"

    heading = table_title
    if header:
        heading += f"\n{table_header}"

    return f"\n{heading}\n\n{instructions}"


def create_page_area(parts_list: List[str]):
    return "\n".join(parts_list)


def convert_df_to_table(
    df: DataFrame,
    columns: Optional[list] = None,
) -> Series:
    """
    Converts a dataframe into a set of instructions to build a table in sphinx.

    Parameters
    ----------
    df : DataFrame
        DataFrame to be converted into a table

    columns: Optional[list]
        List of columns to be converted to table

    header: bool
        Include header in the rows

    Returns
    -------
    Series
        A series in which each entry is a string expressing the contruction in
        rst format of the respective row.
    """
    if columns is None:
        columns = df.columns

    columns_df = DataFrame(df.columns).T
    columns_df.columns = df.columns
    df = concat([columns_df, df], ignore_index=True)

    to_transform = [df[x] for x in columns]
    df["sphinx"] = ["\n\t  - ".join(i) for i in zip(*to_transform)]
    df["sphinx"] = "\t* - " + df.sphinx

    return "\n".join(list(df.sphinx))
