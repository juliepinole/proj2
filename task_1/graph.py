import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


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
        axs[i,j].set_xlabel('Time (milliseconds)', fontsize = fontsize['xlabel'])
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
    fontsize: dict = {
        'ax_title': 12,
    },
    custom_bins: dict = None,
    rename_cols_for_title = None,
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
            axs[i,j].bar(
                series_frequency.index,
                series_frequency.values,
                label=series_frequency.index,
            )
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
