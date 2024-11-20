import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import py_lib as lib


def plot_multiple_heartbeats(
    df: pd.DataFrame(),
    figsize: tuple = (7, 7),
    ncols: int = 2,
    custom_idx_to_plot: list = None,
    n_random_idx: int = 4,
    suptitle: str = 'Heartbeats of patients with outcome ',
    fontsize: dict = {
        'ax_title': 8,
        'xlabel': 8,
    },
    ):

    if custom_idx_to_plot is not None:
        n_plots = len(custom_idx_to_plot)
        final_idx_to_plot = custom_idx_to_plot
    else:
        n_plots = n_random_idx
        possible_idx = list(df.index)
        final_idx_to_plot = random.sample(possible_idx, n_random_idx)
    if n_plots < 2:
        raise ValueError('n_plots should at least be 2 and instead it is {}'.format(n_plots))

    nrows, r = divmod(n_plots, ncols)
    if r > 0:
        nrows += 1
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
    )
    for idx_plt, idx_htb in enumerate(final_idx_to_plot):
        i, j = divmod(idx_plt, ncols)
        axs[i,j].plot(df.loc[idx_htb, :])
        axs[i,j].set_xlabel('Time', fontsize = fontsize['xlabel'])
        axs[i,j].set_title(f'Index Heartbeat {idx_htb}', fontsize = fontsize['ax_title'])

    fig.suptitle(suptitle)
    plt.tight_layout()



def plot_multiple_heartbeats_by_outcome(
    df: pd.DataFrame(),
    outcome = 1.0,
    col_for_outcome = 187,
    figsize: tuple = (7, 7),
    ncols: int = 2,
    custom_idx_to_plot: list = None,
    n_random_idx: int = 4,
    fontsize: dict = {
        'ax_title': 8,
        'xlabel': 8,
    },
    ):
    df_filtered_outcome = df[df[col_for_outcome] == outcome].copy()
    df_filtered_outcome = df_filtered_outcome.drop(columns=[col_for_outcome])
    plot_multiple_heartbeats(
        df_filtered_outcome,
        figsize=figsize,
        ncols=ncols,
        custom_idx_to_plot=custom_idx_to_plot,
        n_random_idx=n_random_idx,
        suptitle=f'Heartbeats of patients with outcome {str(outcome)}',
        fontsize=fontsize,
    )


def dist_mult_plots(
    df: pd.DataFrame(),
    cols: list = ['Age'],
    figsize: tuple = (7, 7),
    ncols: int = 2,
    bar_plot: bool = False,
    rotate_bar_ticks: int = None,
    fontsize: dict = {
        'ax_title': 12,
    },
    custom_bins: dict = None,
    rename_cols_for_title = None,
    custom_messages = None,
    vert_placement_corr: float = 0.8,
    **kwargs,
    ):
    n_var = len(cols)
    nrows, r = divmod(n_var, ncols)
    if r > 0:
        nrows += 1
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
    )
    for item, col in enumerate(cols):
        i, j = divmod(item, ncols)
        if bar_plot:
            to_plot = df[col].astype(str) if is_numeric_dtype(df[col]) else df[col]
            series_frequency = to_plot.value_counts(dropna=False)
            bars = axs[i,j].bar(
                series_frequency.index,
                series_frequency.values,
                label=series_frequency.index,
            )
            if custom_messages is not None:
                custom_messages_list = []
                for lab in series_frequency.index:
                    custom_messages_list.append(f'{custom_messages[lab]:.0%}')
                for bar, message in zip(bars, custom_messages_list):
                    height = bar.get_height()  # Get the height of the bar
                    axs[i,j].text(
                        bar.get_x() + bar.get_width() / 2,  # x-coordinate: center of the bar
                        height*vert_placement_corr,  # y-coordinate: top of the bar
                        message,  # The custom message
                        ha='center',  # Horizontal alignment
                        va='bottom',  # Vertical alignment
                        fontweight='bold',
                        color='w'
                    )                
            if rotate_bar_ticks is not None:
                axs[i,j].tick_params(axis='x', labelrotation=rotate_bar_ticks)
        else:
            bins = 'auto'
            if custom_bins is not None and col in custom_bins:
                bins = custom_bins[col]
            sns.histplot(
                data=df,
                x=col,
                ax=axs[i,j],
                bins=bins,
                kde=True,
                common_bins=False,
                common_norm=False,
            )
    
        title_ax = col
        if rename_cols_for_title is not None:
            if col in rename_cols_for_title:
                title_ax = rename_cols_for_title[col]

        axs[i,j].set_title(
            title_ax,
            fontsize=fontsize['ax_title'],
        )
        axs[i,j].set_xlabel('')
        axs[i,j].set_ylabel('')

    while j < ncols - 1:
        j += 1
        axs[i,j].set_visible(False)
    fig.tight_layout()
    fig.show()


def plot_superposed_heartbeats(
    epochs_dict: dict,
    figsize: tuple = (7, 7),
    title: str = 'Superposed Heartbeats',
    fontsize: dict = {
        'ax_title': 12,
        'xlabel': 8,
    },
    sampling_epochs: int = 10,
    by_outcome: str = None,
    idx_by_class: dict = None,
    ):
    fig, ax = plt.subplots(figsize=figsize)
    possible_idx = list(epochs_dict.keys())

    # Keeping only the indexes of the outcome of interest is by_outcome is specified.
    if by_outcome is not None:
        possible_idx = [idx for idx in possible_idx if idx in idx_by_class[by_outcome]]
        title = title + ' by outcome ' + str(by_outcome)
    
    # Sampling some heartbeats if sampling_epochs is specified.
    if sampling_epochs is not None:
        n_samples = min(sampling_epochs, len(epochs_dict))
        final_idx_to_plot = random.sample(possible_idx, n_samples)
    else:
        final_idx_to_plot = possible_idx

    # Plotting
    for idx, signal in epochs_dict.items():
        if idx in final_idx_to_plot:
            ax.plot(signal)
    ax.set_xlabel('Time', fontsize = fontsize['xlabel'])
    ax.set_title(title, fontsize = fontsize['ax_title'])
    fig.show()


def plot_mean_sd_heartbeats(
    ecg_cleaned_df: pd.DataFrame,
    figsize: tuple = (7, 7),
    title: str = 'Mean & Standard Deviation Heartbeats',
    fontsize: dict = {
        'ax_title': 12,
        'xlabel': 8,
    },
    by_outcome: str = None,
    idx_by_class: dict = None,
    sd_magnitude: dict = {
        '68%': {'factor': 1, 'alpha': 0.5},
         '95%': {'factor': 2, 'alpha': 0.25},
    },
):
    fig, ax = plt.subplots(figsize=figsize)
    # possible_idx = list(ecg_cleaned_df.index)

    # Keeping only the indexes of the outcome of interest is by_outcome is specified.
    if by_outcome is not None:
        # possible_idx = [idx for idx in possible_idx if idx in idx_by_class[by_outcome]]
        title = title + ' by outcome ' + str(by_outcome)
        ecg_cleaned_df_aux = ecg_cleaned_df.loc[ecg_cleaned_df.index.isin(idx_by_class[by_outcome])].copy()
    else:
        ecg_cleaned_df_aux = ecg_cleaned_df.copy()
    
    # Computing mean and standard deviation
    ecg_mean_sd = lib.mean_and_sd_df(ecg_cleaned_df_aux)

    # Plotting the mean
    ax.plot(ecg_mean_sd.columns, ecg_mean_sd.loc['mean', :], label='Mean')

    # Plotting the standard deviation intervals
    for key, value in sd_magnitude.items():
        ecg_mean_sd.loc['upper', :] = ecg_mean_sd.loc['mean', :] + value['factor'] * ecg_mean_sd.loc['sd', :]
        ecg_mean_sd.loc['lower', :] = ecg_mean_sd.loc['mean', :] - value['factor'] * ecg_mean_sd.loc['sd', :]
        ax.fill_between(
            ecg_mean_sd.columns,
            ecg_mean_sd.loc['upper', :],
            ecg_mean_sd.loc['lower', :],
            alpha=value['alpha'],
            label='Standard Deviation ' + key,
        )

    ax.set_xlabel('Time', fontsize = fontsize['xlabel'])
    ax.set_title(title, fontsize = fontsize['ax_title'])
    fig.show()

##########################################################################
# Evaluation
##########################################################################

# Inspired from https://github.com/yohann84L/plot_metric/blob/master/plot_metric/functions.py
def custom_roc_curve(
        fpr,
        tpr,
        thresholds_values,
        thresholds_to_plot: list = [0.5],
        figsize: tuple = (7, 7),
        fontsize: dict = {
            'ax_title': 12,
        },
        dataset_type: str = 'train',
        auc_performance: float = None,
        c_roc_curve='black',
        c_random_guess='red',
        c_thresh_lines='black',
        ls_roc_curve='-',
        ls_thresh_lines=':',
        ls_random_guess='--',
        loc_legend='lower right',
        plot_threshold=True, linewidth=2, y_text_margin=0.05, x_text_margin=0.2,
):

    fig, ax = plt.subplots(figsize=figsize, squeeze=True)
    ax.set_title(
        f'ROC curve, AUC = {auc_performance:.2f}, dataset = {dataset_type}', fontsize=fontsize['ax_title']
        )

    # Plot roc curve
    ax.plot(fpr, tpr, color=c_roc_curve,
                lw=linewidth, label=f'ROC curve (area = {auc_performance:.2f})', linestyle=ls_roc_curve)

    # Plot reference line
    ax.plot([0, 1], [0, 1], color=c_random_guess, lw=linewidth, linestyle=ls_random_guess, label="Random guess")

    # Plot Thresholds
    if plot_threshold:
        for t in thresholds_to_plot:
            # Compute the y & x axis to trace the threshold
            idx_thresh, idy_thresh = (
                fpr[np.argmin(abs(thresholds_values - t))],
                tpr[np.argmin(abs(thresholds_values - t))]
            )
            # Plot vertical and horizontal line
            plt.axhline(y=idy_thresh, color=c_thresh_lines, linestyle=ls_thresh_lines, lw=linewidth)
            plt.axvline(x=idx_thresh, color=c_thresh_lines, linestyle=ls_thresh_lines, lw=linewidth)
            # Plot text threshold
            if idx_thresh > 0.5 and idy_thresh > 0.5:
                plt.text(x=idx_thresh - x_text_margin, y=idy_thresh - y_text_margin,
                            s='Threshold : {:.1f}'.format(t))
            elif idx_thresh <= 0.5 and idy_thresh <= 0.5:
                plt.text(x=idx_thresh + x_text_margin, y=idy_thresh + y_text_margin,
                            s='Threshold : {:.1f}'.format(t))
            elif idx_thresh <= 0.5 < idy_thresh:
                plt.text(x=idx_thresh + x_text_margin, y=idy_thresh - y_text_margin,
                            s='Threshold : {:.1f}'.format(t))
            elif idx_thresh > 0.5 >= idy_thresh:
                plt.text(x=idx_thresh - x_text_margin, y=idy_thresh + y_text_margin,
                            s='Threshold : {:.1f}'.format(t))

            # Plot redpoint of threshold on the ROC curve
            plt.plot(idx_thresh, idy_thresh, 'ro')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    # manipulate
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals], fontsize=9)
    vals = ax.get_xticks()
    ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals], fontsize=9)
    ax.legend(loc=loc_legend)
