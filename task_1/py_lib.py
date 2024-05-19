import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import neurokit2 as nk
import custom_ecg_delineate as custom
import tsfresh as tsf
import os 
import glob
import torch
# import torch.nn.functional as F
from collections import namedtuple
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def pre_process_ecg(
        df: pd.DataFrame,
        label_col_pos: int = -1,
        test_size: float = 0.2,
        random_state: int = 42,
        split_data: bool = True,
        with_lstm_transfo: bool = False,
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
    
    if with_lstm_transfo:
        lstm_input = {}
        for key, data_item in {
            'features': {
                'x_train': x_train_0,
                'x_test': x_test_0,               
            },
            'labels': {
                'y_train': y_train_0,
                'y_test': y_test_0,                
            },
            }.items():
            for second_key, data in data_item.items():
                if key == 'features':
                    if data is not None:
                        data_array = data.to_numpy()
                        data_array = data_array.reshape([data.shape[0], data.shape[1], 1])
                        lstm_input[second_key] = data_array
                if key == 'labels':
                    if data is not None:
                        data_array = data.to_numpy()
                        data_array = data_array#.flatten()
                        lstm_input[second_key] = data_array
        return x_train_0, x_test_0, y_train_0, y_test_0, lstm_input
    else:
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
        dir_path: str = '../output/cs_files_tsfresh/extracted_features',
):
    for i in range(n_loops):
        extracted_features_final = tsfresh_features_extraction(
            x_train_0_transfo,
            starting_point=starting_point,
            window_size=window_size,
        )
        extracted_features_final.to_csv(
            f'{dir_path}_{str(starting_point)}_{str(starting_point+window_size-1)}.csv',
              index=True
              )
        starting_point += window_size


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

#########################################################################
# ML Functions
#########################################################################

def pre_process_features(
        df: pd.DataFrame,
        num_features: list,
        categorical_features: list,
        label_col: str = 'HeartDisease',
        add_one_hot_encoded: bool = False,
        add_embeddings: bool = False,
        stand_features: bool = False,
        stand_embeddings: bool = False,
        test_size: float = 0.2,
        random_state: int = 42,
        category_to_drop: dict = None,
        split_data: bool = True,
        max_emb_dim: int = 50,
        replace_pb_values: dict = {
            'Cholesterol': {
                'target_to_replace': 0,
                'replacement_method': 'median',
            }
        },
):
    """
    Pre-process the features of the dataset.
    :param df: pd.DataFrame: the dataset
    :param num_features: list: numerical features
    :param categorical_features: list: categorical features
    :param label_col: str: the label column
    :param add_one_hot_encoded: bool: whether to add one-hot encoded columns
    :param stand_features: bool: whether to standardize features
    :param test_size: float: test size
    :param random_state: int: random state
    :param category_to_drop: dict: category to drop
    :param split_data: bool: whether to split data into train and test
    :return: namedtuple: train_test_results, categorical_features
    """

    if add_embeddings and add_one_hot_encoded:
        raise ValueError("You cannot add embeddings and one-hot encoded columns at the same time.")
    
    df_aux = df.copy()
    # Step -1: Replace problematic values
    if replace_pb_values is not None:
        for feature, dic_feat in replace_pb_values.items():
            if dic_feat['replacement_method'] == 'median':
                subset_for_comput = df_aux[feature].copy()
                pb_val = dic_feat['target_to_replace']
                print(
                    f'Before: Number of rows with problematic value: {subset_for_comput[subset_for_comput == pb_val].shape}'
                    )
                no_pb_rows = subset_for_comput[subset_for_comput != dic_feat['target_to_replace']]
                replacement_value = np.median(no_pb_rows).astype(int)
                df_aux[feature] = df_aux[feature].replace({dic_feat['target_to_replace']: replacement_value})
                print(
                    f'After: Number of rows with problematic value: {df_aux[feature][df_aux[feature] == pb_val].shape}'
                    )


    # Step 0: Encode categorical features
    label_encoders = {}
    emb_dims = []
    if add_embeddings:
        print('adding embeddings')
        df_aux, label_encoders = encode_categorical_features(df_aux, categorical_features)
        df_x = df_aux[num_features + categorical_features].copy()
        all_features = num_features + categorical_features
        cat_dims = [int(df_x[col].nunique()) for col in categorical_features]
        # TODO(jpinole): check if this is the best way to calculate the embedding dimensions
        emb_dims = [(x, min(max_emb_dim, (x + 1) // 2)) for x in cat_dims]
        df_x_cat = df_x[categorical_features]
        df_x_num = df_x[num_features].astype(np.float32)
    
    # Step 1: Adding One-Hot_Encoded columns
    elif add_one_hot_encoded:
        print('adding One Hot Encoded')
        df_x, categorical_features, all_features = adding_one_hot_encoded(
            df_aux[num_features + categorical_features],
            categorical_features,
            num_features = num_features,
            category_to_drop=category_to_drop,
            drop_first=False,
            )
        df_x_cat = None
        df_x_num = df_x.astype(np.float32) # After OHE, categorical features have become numerical

    else:
        df_x = df[num_features].copy()
        df_x_cat = None
        df_x_num = df_x.astype(np.float32).copy()
        all_features = num_features

    # Step 2: Standardizing features
    if stand_features:
        df_x_num = standardize_features(df_x_num, cols_to_standardize=df_x_num.columns)
    if stand_embeddings:
        df_x_cat = standardize_features(df_x_cat, cols_to_standardize=df_x_cat.columns)
    if df_x_cat is not None:
        df_x = pd.concat([df_x_num, df_x_cat], axis=1)
    else:
        df_x = df_x_num
    
    # Step 3: Split test/ train
    if split_data:
        x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(
        df_x, df[label_col], test_size=test_size, random_state=random_state)
        print(list(x_train_0.columns))
    else:
        x_train_0 = df_x
        y_train_0 = df[label_col]
        print(list(x_train_0.columns))
    
    # Organize data in a dictionary
    if split_data:
        data_dataframes = {
            'X_train': {
                'num': x_train_0[num_features],
                'cat': x_train_0[categorical_features],
                'all': x_train_0,  
            },
            'X_test': {
                'num': x_test_0[num_features],
                'cat': x_test_0[categorical_features],
                'all': x_test_0,  
            },
            'y_train': y_train_0,
            'y_test': y_test_0,
        }
    else:
        data_dataframes = {
            'X_train': {
                'num': x_train_0[num_features],
                'cat': x_train_0[categorical_features],
                'all': x_train_0,  
            },
            'y_train': y_train_0,
        }


    # Step 4: Create Tensors
    x_train_dict = {}
    x_test_dict = {}
    y_train = torch.Tensor(y_train_0.to_numpy())
    for features_type, df in data_dataframes['X_train'].items():
        # x_train_dict[features_type] = torch.Tensor(df.to_numpy())
        x_train_dict[features_type] = torch.from_numpy(df.to_numpy())

    if split_data:
        y_test = torch.Tensor(y_test_0.to_numpy())
        for features_type, df in data_dataframes['X_test'].items():
            x_test_dict[features_type] = torch.from_numpy(df.to_numpy())
        data_tensors = {
            'X_train': x_train_dict,
            'X_test': x_test_dict,
            'y_train': y_train,
            'y_test': y_test,
        }
    else:
        data_tensors = {
            'X_train': x_train_dict,
            'y_train': y_train,
        }

    # Step 5: Store the results in a named tuple
    Train_test_results = namedtuple("train_test_results", "dataframes tensors")
    train_test_results = Train_test_results(
        dataframes=data_dataframes,
        tensors=data_tensors,
        )
    return (
        train_test_results,
        categorical_features,
        all_features,
        {'label_encoders': label_encoders, 'emb_dims': emb_dims}
        )


def process_for_eval_from_single_proba_array(
    y_pred_array: np.array,
):
    y_pred_tensor = torch.from_numpy(y_pred_array)
    y_pred_round_tensor = y_pred_tensor.round()
    y_pred_round = y_pred_round_tensor.detach().numpy()
    return y_pred_tensor, y_pred_round, y_pred_round_tensor


def encode_categorical_features(
        df: pd.DataFrame,
        categorical_features: list,
) ->tuple[pd.DataFrame,dict]:
    '''
    Encode the categorical features of the dataset.
    :param df: pd.DataFrame: the dataset
    :param categorical_features: list: categorical features
    :return: pd.DataFrame: encoded dataset
    '''
    label_encoders = {}
    for cat_col in categorical_features:
        # TODO(jpinole): is it ok to use an encoder designed for labels and not features?
        label_encoders[cat_col] = LabelEncoder()
        df[cat_col] = label_encoders[cat_col].fit_transform(df[cat_col])
    return df, label_encoders


def standardize_features(
    df: pd.DataFrame,
    cols_to_standardize: list = None,
) -> pd.DataFrame:
    '''
    Standardize the features of the dataset.
    :param df: pd.DataFrame: the dataset
    :param cols_to_standardize: list: columns to standardize
    :return: pd.DataFrame: standardized dataset
    '''
    scaler = MinMaxScaler()
    if cols_to_standardize is None:
        untouched_columns = list(df.columns)
    else:
        untouched_columns = [col for col in df.columns if col not in cols_to_standardize]
    df_standardized = scaler.fit_transform(df[cols_to_standardize].copy())
    return pd.concat(
        [df[untouched_columns], pd.DataFrame(df_standardized, columns=cols_to_standardize)],
        axis=1
        )

def adding_one_hot_encoded(
        df: pd.DataFrame,
        cols_obj_pure: list,
        num_features: list = None,
        category_to_drop: dict = None,
        drop_first: bool = False,
        # drop_one_category: bool = False,
        ):
    # We set drop_first=False when we want to select manually which dummy column to drop, to have a benchmark that makes sense.
    df_hot = pd.get_dummies(df, columns=cols_obj_pure, drop_first=drop_first)
    print(list(df_hot.columns))
    if category_to_drop is not None:
        for var, category in category_to_drop.items():
            df_hot.drop('_'.join([var, category]), axis=1, inplace=True)
    all_features = list(df_hot.columns)
    categorical_features = list(set(all_features) - set(num_features))
    return df_hot, categorical_features, all_features


def some_features_selection(
    df: pd.DataFrame,
    thresholds_n_nan=100,
):
    df_aux = df.copy()
    # Drop columns with too many Nan
    count_nan = df.isna().sum()
    cols_to_drop = count_nan[count_nan > thresholds_n_nan].index
    df_aux = df_aux.drop(columns=cols_to_drop)

    # Drop columns with only one value (non-informative)
    features_infos = features_information(df_aux)
    cols_to_drop_non_informative = list(features_infos[features_infos['cardinality'] < 2].index)
    df_aux = df_aux.drop(columns=cols_to_drop_non_informative)
    return df_aux


def features_information(df):
    df_val_count = pd.DataFrame(columns = df.columns, index=['cardinality'])
    for col in df.columns:
        df_val_count.loc['cardinality', col] = len(df[col].value_counts())
    df_val_count = df_val_count.T
    df_val_count['cardinality'] = pd.to_numeric(df_val_count['cardinality'])
    return df_val_count.sort_values(by=['cardinality'], ascending=False)

