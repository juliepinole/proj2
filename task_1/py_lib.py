import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import neurokit2 as nk
import custom_ecg_delineate as custom
import tsfresh as tsf

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


def heartbeats_other_peaks_extraction_loop(
        ecg_cleaned_df: pd.DataFrame, # Better if index already sorted
        r_peaks_df: pd.DataFrame,
        sampling_rate: int = 125,
        starting_point: int = 0,
        window_size: int = 10,
        n_loops: int = 10,
):
    for i in range(n_loops):
        # Building the subsets df
        ecg_subset = ecg_cleaned_df.iloc[starting_point:starting_point+window_size, :].copy()
        ecg_subset_idx = ecg_subset.index
        r_peaks_subset = r_peaks_df.loc[ecg_subset_idx].copy()
        #Extracting the other peaks
        (waves_sum_df, peak_index_df, idx_multiple_peaks_waves) = heartbeats_other_peaks(
            ecg_subset,
            r_peaks_subset,
            sampling_rate=sampling_rate,
        )
        waves_sum_df.to_csv(
            f'../output/heartbeats_extracted/waves/waves_{str(starting_point)}_{str(starting_point+window_size-1)}.csv',
              index=True
              )
        peak_index_df.to_csv(
            f'../output/heartbeats_extracted/peak_index/peak_index_{str(starting_point)}_{str(starting_point+window_size-1)}.csv',
              index=True
              )
        starting_point += window_size

def download_other_peaks_csv_batches(
    dir_path: str = '../output/heartbeats_extracted/',
    types_of_files=['waves', 'peak_index'],
    pull_all: bool = False,
    starting_point: int = 0,
    window_size: int = 10,
    n_loops: int = 10,
):
    # Initialize
    df_placeholder = {}
    
    # Pull all what is in the directory (option 1)
    if pull_all:
        for type_of_file in types_of_files:
            df_placeholder[type_of_file] = []
            csv_files = glob.glob(os.path.join(dir_path, type_of_file, "*.csv"))
            print(os.path.join(dir_path, type_of_file, "*.csv"))
            # loop over the list of csv files 
            for f in csv_files: 
                print(f)
                # read the csv file 
                df_placeholder[type_of_file].append(pd.read_csv(f, index_col=['heartbeat_idx']))

    # # Pull custom files (option 2)
    # else:
    #     for i in range(n_loops):
    #         extracted_features_subset = pd.read_csv(
    #             f'{dir_path}{str(starting_point)}_{str(starting_point+window_size-1)}.csv',
    #             index_col=0,
    #         )
    #         print(f'{dir_path}{str(starting_point)}_{str(starting_point+window_size-1)}.csv')
    #         df_placeholder.append(extracted_features_subset)
    #         starting_point += window_size
    final_df = {}
    for type_of_file in types_of_files:
        final_df[type_of_file] = pd.concat(df_placeholder[type_of_file], axis=0)
        # final_df.index.name='heartbeat_idx'
        final_df[type_of_file] = final_df[type_of_file].sort_index()
    return final_df


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


def tsfresh_features_extraction(
        x_train_0_transfo: pd.DataFrame,
        starting_point: int = 0,
        window_size: int = 100,
):
    x_train_subset = x_train_0_transfo.iloc[:,starting_point:starting_point+window_size].copy()
    x_train_subset['id'] = 1
    extracted_features = tsf.extract_features(x_train_subset, column_id='id')
    columns_list = list(extracted_features.columns)

    heartbeat_idx_placeholder = []
    metrics_by_htb_placeholder = {}
    df_heartbeats_placeholder = []
    for col_name in columns_list:
        underscore_pos = col_name.find('__')
        htb_idx = col_name[:underscore_pos]
        heartbeat_idx_placeholder.append(htb_idx)
        metric_df = extracted_features[[col_name]]
        metric_df = metric_df.rename(columns={col_name: col_name[underscore_pos+2:]})
        metric_df.index = [htb_idx]
        if htb_idx in metrics_by_htb_placeholder:
            metrics_by_htb_placeholder[htb_idx].append(metric_df)
        else:
            metrics_by_htb_placeholder[htb_idx] = [metric_df]
    set_heartbeat_idx = set(heartbeat_idx_placeholder)

    # Concatenate the Dataframes of each heartbeat
    for htb_idx in set_heartbeat_idx:
        df_heartbeats_placeholder.append(
            pd.concat(metrics_by_htb_placeholder[htb_idx], axis=1)
        )
    
    # Concatenate to obtain one dataframe
    extracted_features_final = pd.concat(df_heartbeats_placeholder, axis=0)
    return extracted_features_final


def tsfresh_features_extraction_loop(
        x_train_0_transfo: pd.DataFrame,
        starting_point: int = 0,
        window_size: int = 10,
        n_loops: int = 10,
):
    for i in range(n_loops):
        extracted_features_final = tsfresh_features_extraction(
            x_train_0_transfo,
            starting_point=starting_point,
            window_size=window_size,
        )
        extracted_features_final.to_csv(
            f'../output/cs_files_tsfresh/extracted_features_{str(starting_point)}_{str(starting_point+window_size-1)}.csv',
              index=True
              )
        starting_point += window_size

import os 
import glob 

def join_all_features_csv_batches(
    dir_path: str = '../output/cs_files_tsfresh/extracted_features_',
    pull_all: bool = False,
    starting_point: int = 0,
    window_size: int = 10,
    n_loops: int = 10,
):
    # Initialize the final dataframe
    df_placeholder = []

    # Pull all what is in the directory (option 1)
    if pull_all:
        csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
        print(os.path.join(dir_path, "*.csv"))
        # loop over the list of csv files 
        for f in csv_files: 
            print(f)
            # read the csv file 
            df_placeholder.append(pd.read_csv(f, index_col=0))

    # Pull custom files (option 2)
    else:
        for i in range(n_loops):
            extracted_features_subset = pd.read_csv(
                f'{dir_path}{str(starting_point)}_{str(starting_point+window_size-1)}.csv',
                index_col=0,
            )
            print(f'{dir_path}{str(starting_point)}_{str(starting_point+window_size-1)}.csv')
            df_placeholder.append(extracted_features_subset)
            starting_point += window_size
    
    final_df = pd.concat(df_placeholder, axis=0)
    final_df.index.name='heartbeat_idx'
    final_df = final_df.sort_index()
    return final_df








