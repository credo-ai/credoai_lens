"""
Sphinx mkd formatting functinality.

This library contains all the utility functions necessacry to format
content in a suitable way for a sphinx rst file.
"""
from typing import List, Optional, Literal
from pandas import DataFrame, Series


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
    if title:
        table_title += f" {title}"

    return f"\n{table_title}\n\n{instructions}"


def create_page(parts_list: List[str]):
    return "\n".join(parts_list)


def convert_df_to_table(df: DataFrame, columns: Optional[list] = None) -> Series:
    """
    Converts a dataframe into a set of instructions to build a table in sphinx.

    Parameters
    ----------
    df : DataFrame
        DataFrame to be converted into a table

    columns: Optional[list]
        List of columns to be converted to table

    Returns
    -------
    Series
        A series in which each entry is a string expressing the contruction in
        rst format of the respective row.
    """
    if columns is None:
        columns = df.columns

    to_transform = [df[x] for x in columns]
    df["sphinx"] = ["\n\t  - ".join(i) for i in zip(*to_transform)]
    df["sphinx"] = "\t* - " + df.sphinx

    return "\n".join(list(df.sphinx))
