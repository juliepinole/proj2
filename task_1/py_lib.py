import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
    df_aux.index.name = 'patient_idx'
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