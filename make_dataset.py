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

def make_dataset2():
    df_scaled = pd.read_csv('./data_minmax_scale.csv')
    unique_combinations = df_scaled[['PI', 'Do', 'intensity']].drop_duplicates()
    test_dataset1_list = []
    test_dataset2_list = []

    attempt = 0
    max_attempts = 100  # 최대 시도 횟수를 정하여 무한 루프 방지

    while len(test_dataset1_list) < 10 and attempt < max_attempts and len(unique_combinations) > 0:
        # 조합을 하나 샘플링하고 unique_combinations에서 제거
        selected_combination = unique_combinations.sample(n=1, random_state=attempt)
        unique_combinations = unique_combinations.drop(selected_combination.index)

        selected_rows = df_scaled[(df_scaled['PI'] == selected_combination.iloc[0]['PI']) & 
                                  (df_scaled['Do'] == selected_combination.iloc[0]['Do']) &
                                  (df_scaled['intensity'] == selected_combination.iloc[0]['intensity'])]
        if len(selected_rows) >= 20:
            sampled_rows = selected_rows.sample(n=20, random_state=1)
            df_scaled = df_scaled.drop(sampled_rows.index)  # 선택된 샘플 제거

            # 데이터셋 #1 준비
            test_df_inputs = sampled_rows[['PI', 'Do', 'intensity', 'exposure_time']]
            test_df_targets = sampled_rows[['cured_height']]
            test_dataset1 = MyDataset(test_df_inputs, test_df_targets)
            test_dataset1_list.append(test_dataset1)

            # 데이터셋 #2 준비
            test_df_inputs = sampled_rows[['intensity', 'exposure_time', 'cured_height']]
            test_df_targets = sampled_rows[['PI', 'Do']]
            test_dataset2 = MyDataset(test_df_inputs, test_df_targets)
            test_dataset2_list.append(test_dataset2)

            if len(test_dataset1_list) >= 10:
                break

        attempt += 1

    # 훈련 데이터셋 준비
    train_df_inputs = df_scaled[['PI', 'Do', 'intensity', 'exposure_time']]
    train_df_targets = df_scaled[['cured_height']]
    train_dataset1 = MyDataset(train_df_inputs, train_df_targets)

    train_df_inputs = df_scaled[['intensity', 'exposure_time', 'cured_height']]
    train_df_targets = df_scaled[['PI', 'Do']]
    train_dataset2 = MyDataset(train_df_inputs, train_df_targets)
    
    return train_dataset1, train_dataset2, test_dataset1_list, test_dataset2_list

def make_dataset3():
    df_scaled = pd.read_csv('./data_minmax_scale.csv')
    unique_combinations = df_scaled[['PI', 'Do', 'intensity']].drop_duplicates()

    max_attempts = 100  # 최대 시도 횟수를 정하여 무한 루프 방지
    attempt = 0

    while attempt < max_attempts and len(unique_combinations) > 0:
        # 조합을 하나 샘플링하고 unique_combinations에서 제거
        selected_combination = unique_combinations.sample(n=1)
        unique_combinations = unique_combinations.drop(selected_combination.index)

        selected_rows = df_scaled[(df_scaled['PI'] == selected_combination.iloc[0]['PI']) &
                                (df_scaled['Do'] == selected_combination.iloc[0]['Do']) &
                                (df_scaled['intensity'] == selected_combination.iloc[0]['intensity'])]

        if len(selected_rows) >= 20:
            df_scaled = df_scaled.drop(selected_rows.index)
            sampled_rows = selected_rows.sample(n=20, random_state=1)

            # 데이터셋 준비
            test_df_inputs = sampled_rows[['PI', 'Do', 'intensity', 'exposure_time']]
            test_df_targets = sampled_rows[['cured_height']]
            test_dataset1 = MyDataset(test_df_inputs, test_df_targets)

            test_df_inputs = sampled_rows[['intensity', 'exposure_time', 'cured_height']]
            test_df_targets = sampled_rows[['PI', 'Do']]
            test_dataset2 = MyDataset(test_df_inputs, test_df_targets)
            break

        attempt += 1

    # 훈련 데이터셋 준비
    train_df_inputs = df_scaled[['PI', 'Do', 'intensity', 'exposure_time']]
    train_df_targets = df_scaled[['cured_height']]
    train_dataset1 = MyDataset(train_df_inputs, train_df_targets)

    train_df_inputs = df_scaled[['intensity', 'exposure_time', 'cured_height']]
    train_df_targets = df_scaled[['PI', 'Do']]
    train_dataset2 = MyDataset(train_df_inputs, train_df_targets)

    # # 겹치는 데이터 검사
    # test_indices = test_df_targets[['PI', 'Do']].reset_index(drop=True)
    # test_indices['intensity'] = test_df_inputs['intensity'].reset_index(drop=True)
    
    # train_indices = train_df_targets[['PI', 'Do']].reset_index(drop=True)
    # train_indices['intensity'] = train_df_inputs['intensity'].reset_index(drop=True)
    
    # common_data = pd.merge(test_indices, train_indices, on=['PI', 'Do', 'intensity'])

    # print("겹치는 데이터가 있습니까?", not common_data.empty)
    # if not common_data.empty:
    #     print("겹치는 데이터:", common_data)

    return train_dataset1, train_dataset2, test_dataset1, test_dataset2
