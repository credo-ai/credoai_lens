from credoai.reporting.plot_utils import (credo_classification_palette, get_axis_size)
from numpy import pi
import matplotlib as mpl
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as sk_metrics

class FairnessReport:
    def __init__(self, toolkit, infographic_shape=(3,5), size=5):
        self.toolkit = toolkit
        self.size = size
        self.infographic_shape = infographic_shape
    
    def create_report(self, include_fairness=True, include_disaggregation=True, filename=None):
        """Creates a fairness report for binary classification model

        Parameters
        ----------
        include_fairness : bool, optional
            Whether to include fairness plots, by default True
        include_disaggregation : bool, optional
            Whether to include performance plots broken down by
            subgroup. Overall performance are always reported, by default True
        filename : string, optional
            If given, the location where the generated pdf report will be saved, by default None
        """        
        df = self.toolkit.get_df()
        # plot
        figs = []
        # comparison plots
        if include_fairness:
            figs.append(self.plot_fairness())
        # individual group performance plots
        figs.append(self.plot_performance(df['true'], df['pred'], 'Overall'))
        if include_disaggregation:
            for group, sub_df in df.groupby('sensitive'):
                figs.append(self.plot_performance(sub_df['true'], sub_df['pred'], group))
            
        if filename is not None:
            pdf = matplotlib.backends.backend_pdf.PdfPages(f"{filename}.pdf")
            for fig in figs: ## will open an empty extra figure :(
                pdf.savefig(fig, bbox_inches='tight', pad_inches=1)
            pdf.close()
            
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
        f, ax = plt.subplots(1,1, figsize=(self.size, self.size))
        self._plot_disaggregated_metrics(ax, self.size)
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
    
    def _plot_disaggregated_metrics(self, ax, size):
        # create df
        sensitive_feature = self.toolkit.sensitive_features.name
        df =  self.toolkit.get_disaggregated_results(False) \
                    .reset_index() \
                    .melt(id_vars=sensitive_feature,
                          var_name='Metric',
                          value_name='Value')
        # plot
        num_cats = len(df[sensitive_feature].unique())
        palette = sns.color_palette('Purples', num_cats)
        palette[-1] = [.4,.4,.4]
        sns.barplot(data=df, 
                     y='Metric', 
                     x='Value', 
                     hue=sensitive_feature,
                     palette=palette,
                     edgecolor='w',
                     ax=ax)

        sns.despine()
        plt.legend(bbox_to_anchor=(1.01, 1.02))
