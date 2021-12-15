import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DEFAULT_STYLE = {'lw': 3,
                 'color': '#4937c0'}

def credo_classification_palette():
    return [[.5,.5,.5], '#3b07b4']
    
def credo_converging_palette(n_colors):\
    # created https://vis4.net/palettes/#/9|s|00429d,96ffea,ffffe0|ffffe0,ff005e,93003a|1|1
    # 12 colors using first and last
    credo_colors = ['#152672', '#382874', '#502a76', '#652b77', '#792c79', '#8c2c7b', '#9f2c7c', '#b12c7e', '#c32b7f', '#d62981', '#e82783', '#fb2384']
    idx = np.round(np.linspace(0, len(credo_colors) - 1, n_colors)).astype(int)
    return [credo_colors[i] for i in idx]

def credo_diverging_palette(n_colors):
    # created https://vis4.net/palettes/#/9|s|00429d,96ffea,ffffe0|ffffe0,ff005e,93003a|1|1
    # 12 colors using first and last
    credo_colors = ['#3b07b4', '#4937c0', '#5257cb', '#5775d7', '#5893e2', '#55b1ed', '#f2c5e2', '#f8acce', '#fb92bb', '#fd75a8', '#fd5596', '#fb2384']
    idx = np.round(np.linspace(0, len(credo_colors) - 1, n_colors)).astype(int)
    return [credo_colors[i] for i in idx]    

def credo_paired_palette(n_colors):
    credo_colors = ['#a6cee3', '#1f78b4', '#ee1d7a', '#ac0047', '#cab2d6', '#6a3d9a',  '#fdbf6f', '#ff7f00']
    return credo_colors[:n_colors]


def get_axis_size(ax, fig):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return [bbox.width, bbox.height]

def outlier_style_df(s):
    '''
    highlight the values above and below 1.5 IQR
    '''
    median = s.median()
    IQR = s.quantile(.75) - s.quantile(.25)
    is_low = s < median-IQR*1.5
    is_high = s > median+IQR*1.5
    low_style = 'font-weight: bold; color: red'
    high_style = 'font-weight: bold; color: blue'
    style = [low_style if l 
             else high_style if h 
             else '' 
             for l, h in zip(is_low, is_high)]
    return style

def plot_curve(x, y, label, 
               ax=None, legend_loc='auto',
               context=None,
              **plot_kwargs):
    style = DEFAULT_STYLE.copy()
    style.update(plot_kwargs)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))

    ax.plot(x, y, label=label, **style)
    if label != "":
        ax.legend(loc=legend_loc)
    return ax


