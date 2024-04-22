import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler
from dataset import *

def inverse_scaler(df, scalers):
    for col in df.columns:
        scaler = scalers[col]
        df[col] = scaler.inverse_transform(df[[col]])
    return df

def apply_scaler(df_new, loaded_scalers):
    for col in df_new.columns:
        scaler = loaded_scalers[col]
        df_new[col] = scaler.transform(df_new[[col]])
    return df_new

def make_dataset():
    df_scaled = pd.read_csv('./data_minmax_scale.csv')

    unique_combinations = df_scaled[['PI', 'Do']].drop_duplicates()
    selected_combination = unique_combinations.sample(n=1)

    selected_rows = df_scaled[(df_scaled['PI'] == selected_combination.iloc[0]['PI']) & (df_scaled['Do'] == selected_combination.iloc[0]['Do'])]

    # 해당 조합에서 무작위로 20개 행 선택 (만약 20개 미만이면 모두 선택)
    sampled_rows = selected_rows.sample(n=min(20, len(selected_rows)), random_state=1)
    test_df = sampled_rows.copy()
    train_df = df_scaled.drop(sampled_rows.index)

    # input: [PI, Do, intensity, exposure time] / output: [cured height]
    train_df_inputs = train_df[['PI', 'Do', 'intensity', 'exposure_time']]
    train_df_targets = train_df[['cured_height']]
    train_dataset1 = MyDataset(train_df_inputs, train_df_targets)

    test_df_inputs = test_df[['PI', 'Do', 'intensity', 'exposure_time']]
    test_df_targets = test_df[['cured_height']]
    test_dataset1 = MyDataset(test_df_inputs, test_df_targets)

    # input: [intensity, exposure time, cured_height] / output: [PI, Do]
    train_df_inputs = train_df[['intensity', 'exposure_time', 'cured_height']]
    train_df_targets = train_df[['PI', 'Do']]
    train_dataset2 = MyDataset(train_df_inputs, train_df_targets)

    test_df_inputs = test_df[['intensity', 'exposure_time', 'cured_height']]
    test_df_targets = test_df[['PI', 'Do']]
    test_dataset2 = MyDataset(test_df_inputs, test_df_targets)
    
    return train_dataset1, train_dataset2, test_dataset1, test_dataset2
