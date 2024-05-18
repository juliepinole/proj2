import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import neurokit2 as nk
import custom_ecg_delineate as custom

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


def heartbeats_other_peaks(
        ecg_cleaned_df: pd.DataFrame,
        r_peaks_df: pd.DataFrame,
        sampling_rate: int = 125,
    ):
    waves_sum_placeholder = []
    peak_index_placeholder = []
    idx_multiple_peaks_waves = []
    for idx, row_ecg_cleaned in ecg_cleaned_df.iterrows():
        # Find the corresponding r_peaks
        row_r_peaks = r_peaks_df.loc[idx]
        # Compute the peaks
        waves , signals = custom.ecg_delineate_custom(
            row_ecg_cleaned,
            row_r_peaks,
            sampling_rate=sampling_rate,
            method='peak',
            )
        # Create the waves_sum content
        waves_sum = waves.sum()
        if waves_sum.values.max() > 1:
            idx_multiple_peaks_waves.append(idx)
        waves_sum.name = idx
        waves_sum_placeholder.append(waves_sum)
        # isolate the index of the peaks
        signals_aux_dict = {}
        for peak_type, peak_index in signals.items():
            signals_aux_dict[peak_type] = peak_index[0]
            signals_df = pd.DataFrame(signals_aux_dict, index=[idx])
        peak_index_placeholder.append(signals_df)
        
    # Concatenate the final dataframes
    waves_sum_df = pd.concat(waves_sum_placeholder, axis=1)
    waves_sum_df = waves_sum_df.T
    waves_sum_df.index.name = 'heartbeat_idx'
    peak_index_df = pd.concat(peak_index_placeholder, axis=0)
    peak_index_df.index.name = 'heartbeat_idx'
    return (waves_sum_df,
            peak_index_df,
            idx_multiple_peaks_waves
            )


def mean_and_sd_df(
        df: pd.DataFrame,
):
    mean_df = df.mean(axis=0).to_frame().T
    mean_df.index = ['mean']
    sd_df = df.std(axis=0).to_frame().T
    sd_df.index = ['sd']
    return pd.concat([mean_df, sd_df])
        

# def peaks_features(
#         all_peaks_df: pd.DataFrame,
#         peaks_names_dict = {
#             'ECG_R_Peaks': 'R',
#             'ECG_S_Peaks': 'S',
#             'ECG_T_Peaks': 'T',
#             'ECG_T_Offsets': 'T_Offsets',
#         },
#         ):
#     # Present the peaks data in a dataframe
#     df_peaks = 

#     # Compute the intervals of interest
#     # df_peaks['QRS_complex'] = 
#     # df_peaks['RS'] = 
#     # df_peaks['PR_interval'] = 
#     # df_peaks['PR_segment'] = 
#     # df_peaks['ST_segment'] = 
#     # df_peaks['QT_interval'] = 

#     return df_peaks


def merge_R_and_other_peaks(
    r_peaks_df,
    peak_index_df,
    other_peaks_to_keep = ['ECG_S_Peaks', 'ECG_T_Peaks', 'ECG_T_Offsets'],
        
):
    # R_peaks
    r_peaks_placeholder = {}
    for heartbeat_idx, r_peak_series in r_peaks_df.iterrows():
        mask = (r_peak_series == 1)
        r_peak_idx = r_peak_series.index[mask]
        r_peaks_placeholder[heartbeat_idx] = r_peak_idx
    all_peaks_df = pd.DataFrame(r_peaks_placeholder, index=['ECG_R_Peaks']).T

    # other peaks
    for peak_type in other_peaks_to_keep:
        all_peaks_df[peak_type] = peak_index_df[peak_type]
    
    return all_peaks_df







