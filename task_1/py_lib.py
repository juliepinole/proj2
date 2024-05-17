import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import neurokit2 as nk

def pre_process_ecg(
        df: pd.DataFrame,
        label_col_pos: int = -1,
        test_size: float = 0.2,
        random_state: int = 42,
        split_data: bool = True,
):
    """
    Pre-process the ecg.
    :param df: pd.DataFrame: the dataset
    :param label_col_pos: int: the position of the column containing the label
    :param test_size: float: test size
    :param random_state: int: random state
    :param split_data: bool: whether to split data into train and test
    :return: x_train_0, x_test_0, y_train_0, y_test_0
    """

    df_aux = df.copy()
    df_aux.index.name = 'heartbeat_idx'
    df_label = df_aux.iloc[:, label_col_pos].to_frame()
    df_time_series = df_aux.drop(df_aux.columns[label_col_pos], axis=1)
    # Step 3: Split test/ train
    if split_data:
        x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(
        df_time_series, df_label, test_size=test_size, random_state=random_state)
    else:
        x_train_0 = df_time_series
        y_train_0 = df_label
        x_test_0 = None
        y_test_0 = None

    return x_train_0, x_test_0, y_train_0, y_test_0


def select_one_row(
        df: pd.DataFrame,
        row_idx: int,
        as_frame: bool = False,
        prefix_column: str = 'row_',
):
    """
    Select one row from the dataframe.
    :param df: pd.DataFrame: the dataset
    :param idx: int: the index of the row to select
    :return: pd.DataFrame: the selected row
    """

    if as_frame:
        df_aux = df.iloc[row_idx,:].copy()
        df_aux.columns = [prefix_column + str(row_idx)]
        return df.iloc[row_idx,:]
    else:
        return df.iloc[row_idx,:]


def create_heartbeats_dictionary(
        df: pd.DataFrame,
        sampling_rate: int = 125,
        method: str = 'neurokit',
    ):
    r_peaks_placeholder = []
    ecg_cleaned_placeholder = []
    epochs_dict = {}
    total_infos_dict = {}
    for idx, row in df.iterrows():
        # Sanitize and clean input
        ecg_signal = nk.signal_sanitize(row.values)
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method=method)
        epochs_dict[idx] = ecg_cleaned
        total_infos_dict[idx] = {
            'signal': ecg_signal,
            'clean_signal': ecg_cleaned,
        }
        ecg_cleaned_placeholder.append(pd.DataFrame(data=ecg_cleaned, columns=[idx]).T)

        # Detect R-peaks
        instant_peaks, info = nk.ecg_peaks(
                ecg_cleaned=ecg_cleaned,
                sampling_rate=sampling_rate,
                method=method,
                correct_artifacts=True,
        )
        total_infos_dict[idx]['r_peaks'] = instant_peaks
        total_infos_dict[idx]['infos'] = info
        instant_peaks = instant_peaks.T
        instant_peaks.index = [idx]
        r_peaks_placeholder.append(instant_peaks)
    
    # Concatenate r_peaks dataframe
    r_peaks_df = pd.concat(r_peaks_placeholder)
    r_peaks_df.index.name = 'heartbeat_idx'

    # Concatenate ecg_clean dataframe
    ecg_cleaned_df = pd.concat(ecg_cleaned_placeholder)
    ecg_cleaned_df.index.name = 'heartbeat_idx'
    return epochs_dict, total_infos_dict, r_peaks_df, ecg_cleaned_df


def mean_and_sd_df(
        df: pd.DataFrame,
):
    mean_df = df.mean(axis=0).to_frame().T
    mean_df.index = ['mean']
    sd_df = df.std(axis=0).to_frame().T
    sd_df.index = ['sd']
    return pd.concat([mean_df, sd_df])
        

