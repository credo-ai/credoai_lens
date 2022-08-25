import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DEFAULT_COLOR = "#4937c0"


def get_style(rc=None, figsize=3, figure_ratio=1, n_cols=1, n_rows=1):
    """Sets default styling for plots

    Figsize is the size of one plot. Fontsizes are also keyed to the fig size.
    If you create a figure with multiple subplots, change the n_cols/n_rows

    See: https://matplotlib.org/stable/tutorials/introductory/customizing.html

    If seaborn is used note that you may have to pass the style to sns.axes_style.
    Example:

    with get_style() as style:
        f, ax = plt.subplots()
        with sns.axes_style(rc=style):
            sns.barplot...

    Parameters
    ----------
    rc : dict
        Dictionary of rc params to override defaults
    figsize : int
        dimension of a single subplot in figure
    figure_ratio : int
        ratio of height/width. >1 will lead to a tall plot. <1 will lead to a wide plot
    n_cols : int
        Set to the number of columns of subplot
    n_rows : int
        Set to the number of rows of subplot

    """
    fig_dims = [figsize * n_cols, figsize * n_rows * figure_ratio]
    # fontsizes defined relative to font.size
    # xx-small, x-small, small, medium, large, x-large, xx-large, larger, or smaller
    style = {
        "figure.figsize": fig_dims,
        "figure.dpi": 150,
        "font.size": figsize * 3,
        "lines.linewidth": figsize / 3,
        "axes.linewidth": figsize / 3,
        "axes.titlesize": "large",
        "axes.titlepad": figsize * 4,
        "axes.labelsize": "medium",
        "axes.labelpad": figsize * 1.5,
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
        "legend.title_fontsize": "x-small",
        "legend.fontsize": "xx-small",
    }
    # adjust the tick params
    tick_pad = figsize
    tick_size = {"major": figsize * 2, "minor": figsize}
    tick_width = {"major": figsize / 4, "minor": figsize / 5}
    for axis in ["ytick", "xtick"]:
        for kind in ["major", "minor"]:
            style[f"{axis}.{kind}.size"] = tick_size[kind]
            style[f"{axis}.{kind}.width"] = tick_width[kind]
            style[f"{axis}.{kind}.pad"] = tick_pad
    if rc:
        style.update(rc)
    return mpl.rc_context(style)


def credo_classification_palette():
    return [[0.5, 0.5, 0.5], "#3b07b4"]


def credo_converging_palette(
    n_colors,
):  # created https://vis4.net/palettes/#/9|s|00429d,96ffea,ffffe0|ffffe0,ff005e,93003a|1|1
    # 12 colors using first and last
    credo_colors = [
        "#152672",
        "#382874",
        "#502a76",
        "#652b77",
        "#792c79",
        "#8c2c7b",
        "#9f2c7c",
        "#b12c7e",
        "#c32b7f",
        "#d62981",
        "#e82783",
        "#fb2384",
    ]
    idx = np.round(np.linspace(0, len(credo_colors) - 1, n_colors)).astype(int)
    return [credo_colors[i] for i in idx]


def credo_diverging_palette(n_colors):
    # created https://vis4.net/palettes/#/9|s|00429d,96ffea,ffffe0|ffffe0,ff005e,93003a|1|1
    # 12 colors using first and last
    credo_colors = [
        "#3b07b4",
        "#4937c0",
        "#5257cb",
        "#5775d7",
        "#5893e2",
        "#55b1ed",
        "#f2c5e2",
        "#f8acce",
        "#fb92bb",
        "#fd75a8",
        "#fd5596",
        "#fb2384",
    ]
    idx = np.round(np.linspace(0, len(credo_colors) - 1, n_colors)).astype(int)
    return [credo_colors[i] for i in idx]


def credo_paired_palette(n_colors):
    credo_colors = [
        "#a6cee3",
        "#1f78b4",
        "#ee1d7a",
        "#ac0047",
        "#cab2d6",
        "#6a3d9a",
        "#fdbf6f",
        "#ff7f00",
    ]
    return credo_colors[:n_colors]


def format_label(metric, wrap_length=15):
    return textwrap.fill(metric.replace("_", " "), wrap_length)


def get_axis_size(ax, fig):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return [bbox.width, bbox.height]


def get_table_style():
    caption_style = {
        "selector": "caption",
        "props": [
            ("text-align", "left"),
            ("font-weight", "bold"),
            ("font-size", "1.25em"),
        ],
    }

    cell_hover = {  # for row hover use <tr> instead of <td>
        "selector": "td:hover",
        "props": [("background-color", "#3b07b4"), ("color", "white")],
    }

    styles = [caption_style, cell_hover]
    return styles


def outlier_style_df(s):
    """
    highlight the values above and below 1.5 IQR
    """
    median = s.median()
    IQR = s.quantile(0.75) - s.quantile(0.25)
    is_low = s < median - IQR * 1.5
    is_high = s > median + IQR * 1.5
    low_style = "font-weight: bold; color: red"
    high_style = "font-weight: bold; color: blue"
    style = [
        low_style if l else high_style if h else "" for l, h in zip(is_low, is_high)
    ]
    return style


def plot_curve(x, y, label, ax=None, legend_loc="auto", context=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))

    ax.plot(x, y, label=label, **kwargs)
    if label != "":
        ax.legend(loc=legend_loc)
    return ax
