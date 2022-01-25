from credoai.reporting.credo_report import CredoReport
from credoai.reporting.plot_utils import (credo_classification_palette, 
                                          format_metric, get_axis_size,
                                          DEFAULT_STYLE)
from numpy import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as sk_metrics

class FairnessReport(CredoReport):
    def __init__(self, module, infographic_shape=(3,5), size=5):
        super().__init__()
        self.module = module
        self.size = size
        self.infographic_shape = infographic_shape
    
    def create_report(self, filename=None, include_fairness=True, include_disaggregation=True):
        """Creates a fairness report for binary classification model

        Parameters
        ----------
        filename : string, optional
            If given, the location where the generated pdf report will be saved, by default None
        include_fairness : bool, optional
            Whether to include fairness plots, by default True
        include_disaggregation : bool, optional
            Whether to include performance plots broken down by
            subgroup. Overall performance are always reported, by default True
            
        Returns
        -------
        array of figures
        """        
        df = self.module.get_df()
        # plot
        # comparison plots
        if include_fairness:
            self.figs.append(self.plot_fairness())
        # individual group performance plots
        self.figs.append(self.plot_performance(df['true'], df['pred'], 'Overall'))
        if include_disaggregation:
            for group, sub_df in df.groupby('sensitive'):
                self.figs.append(self.plot_performance(sub_df['true'], sub_df['pred'], group))
        # save
        if filename is not None:
            self.export_report(filename)
        return self.figs

    def plot_fairness(self):
        """Plots fairness for binary classification

        Creates plots for binary classification model that summarizes
        performance disparities across groups. Individual group
        performance plots are also relevant for fully describing
        performance differences.

        Returns
        -------
        matplotlib figure
        """
        plot_disaggregated = False
        if self.module.metric_frames != {}:
            plot_disaggregated = True
        n_plots = 1+plot_disaggregated
        with sns.plotting_context('talk', font_scale=self.size/7):
            f, ax = plt.subplots(1, n_plots, figsize=(self.size*n_plots, 
                                                       self.size))
        plt.subplots_adjust(wspace=0.5)
        axes = f.get_axes()
        # plot fairness
        self._plot_fairness_metrics(axes[0], self.size)
        if plot_disaggregated:
            self._plot_disaggregated_metrics(axes[1], self.size)
        return f
    
    def plot_performance(self, y_true, y_pred, label, **grid_kwargs):
        """Plots performance for binary classification

        Plots "infographic" depiction of outcomes for ground truth
        and model performance, as well as a confusion matrix.

        Parameters
        ----------
        y_true : (List, pandas.Series, numpy.ndarray)
            The ground-truth labels (for classification) or target values (for regression).
        y_pred : (List, pandas.Series, numpy.ndarray)
            The predicted labels for classification
        label : str
            super title for set of performance plots

        Returns
        -------
        matplotlib figure
        """        
        true_data, pred_data = self._create_data(y_true, y_pred)
        # plotting
        ratio = self.infographic_shape[0]/self.infographic_shape[1]
        f, [true_ax, pred_ax, confusion_ax] = plt.subplots(1,3, figsize=(self.size*3, 
                                                                         self.size*ratio))
        self._plot_grid(true_data, true_ax, self.size, **grid_kwargs)
        self._plot_grid(pred_data, pred_ax, self.size, **grid_kwargs)
        self._plot_confusion_matrix(y_true, y_pred, confusion_ax, self.size/2)
        # add text
        true_ax.set_title('Ground Truth', fontsize=self.size*3, pad=0)
        pred_ax.set_title('Model Predictions', fontsize=self.size*3, pad=0)
        confusion_ax.set_title('Confusion Matrix', fontsize=self.size*2)
        for ax, rate in [(true_ax, y_true.mean()), (pred_ax, y_pred.mean())]:
            ax.text(.5, 0, f'Positive Rate = {rate: .2f}', 
                    fontsize=self.size*2, transform = ax.transAxes, ha='center')

        # other text objects
        text_objects = [
            (.5, 1, label, {'fontweight': 'bold', 'fontsize': self.size*6})
        ]
        text_ax = self._plot_text(f, text_objects)
        return f
        

        
    def _create_data(self, y_true, y_pred):
        n = self.infographic_shape[0] * self.infographic_shape[1]
        true_pos_n = int(np.mean(y_true)*n)
        pred_pos_n = int(np.mean(y_pred)*n)
        true_data = np.reshape([1]*true_pos_n + [0]*(n-true_pos_n), self.infographic_shape)
        pred_data = np.reshape([1]*pred_pos_n + [0]*(n-pred_pos_n), self.infographic_shape)
        return true_data, pred_data

    def _plot_circles(self,
                      data, 
                      ax, colors, marker='o'):
        n_rows, n_cols = data.shape

        # set up point locations
        y, x = np.unravel_index(np.arange(data.size), data.shape)
        y = y + .5
        x = x + .5

        #set up limits
        # final touches
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim([-.5, n_cols+.5])
        ax.set_ylim([-.5, n_rows+.5])
        ax.invert_yaxis()

        # set up size
        # radius in data coordinates:
        r = 0.4
        # radius in display coordinates:
        r_ = ax.transData.transform([r,0])[0] - ax.transData.transform([0,0])[0]
        # marker size as the area of a circle
        markersize = pi * r_**2

        # plotting
        ax.scatter(x, y, 
                   marker=marker,
                   s = markersize, 
                   c = colors)
        sns.despine(left=True, bottom=True)

    def _plot_grid(self,
                   data, 
                   ax,
                   size,
                   marker='circle',
                   palette=credo_classification_palette(),
                   sort='sorted'):
        n_rows, n_cols = data.shape
        # set up positive groups
        if sort == 'shuffle':
            np.random.shuffle(data)
        elif sort == 'sorted':
            data = np.reshape(np.sort(data.flatten()), data.shape)
        # plot figure
        if marker == 'circle':
            colors = np.array([palette[i] for i in data.flatten()], dtype=object)
            self._plot_circles(data, ax, colors)
        else:
            sns.heatmap(data, cbar=False, linewidth=3, cmap=palette, ax=ax)
        # success plotting
        success_colors = [[1,1,1,0], [1,1,1,1]]
        success_colors = np.array([success_colors[i] for i in data.flatten()], dtype=object)
        self._plot_circles(data, ax, success_colors, marker='$\checkmark$')

        # failure plotting
        failure_colors = [[0,0,0], [1,1,1,0]]
        failure_colors = np.array([failure_colors[i] for i in data.flatten()], dtype=object)
        self._plot_circles(data, ax, failure_colors, marker='x')    

        ax.set_xticks([])
        ax.set_yticks([])
        return ax
    
    def _plot_confusion_matrix(self, y_true, y_pred, ax, size):
        mat = sk_metrics.confusion_matrix(y_true, y_pred, normalize='true')
        sns.heatmap(mat, 
                    square=True, 
                    cbar=False,
                    linewidth=size/2,
                    xticklabels = ['Negative', 'Positive'],
                    annot=True,
                    fmt='.1%',
                    cmap='Purples',
                    annot_kws={'fontsize': size*6},
                    ax=ax
                   )
        ax.set_yticklabels(['Negative', 'Positive'], va='center', rotation = 90, position=(0,0.28))
        ax.tick_params(labelsize=size*5, length=0, pad=size/2)
        # labels
        ax.text(-.15, .5, 'Ground Truth', fontsize=size*5,
                va='center', rotation = 90, transform=ax.transAxes, fontweight='bold')
        ax.text(.5, -.15, 'Prediction', fontsize=size*5,
                ha='center', transform=ax.transAxes, fontweight='bold')

        # TPR, FPR, labels
        labels = 'TN', 'FN', 'FP', 'TP'
        locations = [(.5, .2), (1.5, .2), (.5, 1.2), (1.5, 1.2)]
        for label, location in zip(labels, locations):
            t = ax.text(location[0], location[1], label, ha='center', fontsize=size*3, color='k')
            t.set_bbox(dict(facecolor='w', alpha=1, edgecolor='white', boxstyle='square,pad=.5'))
        
    def _plot_text(self, f, text_objects):
        ax = f.add_axes([0,0,1,1])
        sns.despine(left=True, bottom=True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_alpha(0.01)
        for x, y, s, kwargs in text_objects:
            ax.text(x, y, s, transform = ax.transAxes, ha='center', **kwargs)
        return ax
    
    def _plot_fairness_metrics(self, ax, size):
        # create df
        df = self.module.get_fairness_results()
        # add parity to names
        df.index = [i+'_parity' if row['kind'] == 'parity' else i
                    for i, row in df.iterrows()]
        df = df['value']
        df.index.name = 'Fairness Metric'
        df.name = 'Value'
        # plot
        sns.barplot(data=df.reset_index(), 
                     y='Fairness Metric', 
                     x='Value',
                     edgecolor='w',
                     color = DEFAULT_STYLE['color'],
                     ax=ax)
        self._style_barplot(ax)
        
    def _plot_disaggregated_metrics(self, ax, size):
        # create df
        sensitive_feature = self.module.sensitive_features.name
        df =  self.module.get_disaggregated_results(False) \
                    .reset_index() \
                    .melt(id_vars=sensitive_feature,
                          var_name='Performance Metric',
                          value_name='Value')
        # plot
        num_cats = len(df[sensitive_feature].unique())
        palette = sns.color_palette('Purples', num_cats)
        palette[-1] = [.4,.4,.4]
        sns.barplot(data=df, 
                     y='Performance Metric', 
                     x='Value', 
                     hue=sensitive_feature,
                     palette=palette,
                     edgecolor='w',
                     ax=ax)
        self._style_barplot(ax)
        plt.legend(bbox_to_anchor=(1.01, 1.02))
        
    def _style_barplot(self, ax):
        sns.despine()
        ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
        ax.set_ylabel(ax.get_xlabel(), fontweight='bold')
        # format metric labels
        ax.set_yticklabels([format_metric(label.get_text()) 
                            for label in ax.get_yticklabels()])
    