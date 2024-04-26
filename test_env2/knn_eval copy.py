import pandas as pd
import torch
import pickle
import numpy as np
import argparse

from collections import Counter
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from make_dataset import *
from dataset import *
from model import ResidualNet
from scipy.optimize import curve_fit
from train import *

def log_regression_knn(make_dataset=3):
    df_scaled = pd.read_csv('./data_minmax_scale.csv')
    df_experiment = pd.read_csv('./data_experiment.csv')
    
    train_df_inputs = df_scaled[['intensity', 'exposure_time', 'cured_height']]
    train_df_targets = df_scaled[['PI', 'Do']]
    train_dataset = MyDataset(train_df_inputs, train_df_targets)
    nearest = NearestNeighbors(n_neighbors=3)
    nearest.fit(train_dataset.input)

    mean_loss = 0
    
    for i in range(10):
        _, train_dataset, _, _ = make_dataset3()

        

        for inputs, targets in test_loader2:
            intensity_values = inputs[:, 0].numpy()  # 'intensity' 추정: 0번째 열
            x_data = inputs[:, 1].numpy()  # 'exposure_time' 추정: 1번째 열
            y_data = inputs[:, 2].numpy()  # 'cured_height' 추정: 2번째 열
            target_data = targets[:, ].numpy()
        
        # 모델 함수 정의
        def model_function(x, a, b, c, d):
            return np.log(b * x + c) / np.log(a) + d    
        
        # 초기 파라미터 추측
        initial_guess = [2, 1, 1, 1]
        params, params_covariance = curve_fit(model_function, x_data, y_data, p0=initial_guess)

        unique_exposure_times = df_origin['exposure_time'].unique()
        x_data_all = np.sort(unique_exposure_times)

        mask = np.isin(x_data_all, x_data)
        close_mask = np.array([np.any(np.isclose(x, x_data)) for x in x_data_all])
        combined_mask = mask | close_mask

        # 근접한 값이나 정확히 일치하는 값을 제외합니다.
        x_data_filtered = x_data_all[~combined_mask]

        # x_data_to_pred_filtered에서 예측을 수행합니다.
        y_pred_filtered = model_function(x_data_filtered, *params)
        print('iter' + str(i))
        
        intensity_repeated = np.repeat(intensity_values[0], len(x_data_filtered))
        new_inputs = np.column_stack((intensity_repeated, x_data_filtered, y_pred_filtered))
        
        if np.isnan(new_inputs).any():
            new_inputs = new_inputs[~np.isnan(new_inputs).any(axis=1)]
        new_inputs_filtered = new_inputs[new_inputs[:, 2] >= 0]
        distances, index = nearest.kneighbors(new_inputs_filtered)
        
        flattened_index = index.flatten()
        
        candidates = train_dataset.target[flattened_index].view(-1, 2)
        candidates_rounded = np.round(candidates, decimals=4)

        # 텐서를 리스트의 튜플로 변환합니다. (numpy 배열을 통해)
        candidates_list = candidates_rounded.numpy().tolist()

        # 각 행의 빈도수 계산
        counter = Counter(tuple(row) for row in candidates_list)

        # 가장 많이 나타난 행 찾기
        most_common_candidate = counter.most_common(1)

        # 가장 많이 나타난 행과 그 빈도수 출력
        most_common_row, frequency = most_common_candidate[0]
        print(most_common_row, target_data[0], frequency)

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(torch.tensor(most_common_row), torch.tensor(target_data[0])).item()
        print("Loss: %f" % (loss))
        mean_loss += loss
    mean_loss /= 10
    print('Mean Loss: %f' % (mean_loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple file processing script.")
    parser.add_argument('--make-dataset', type=int, default=3)
    
    args = parser.parse_args()
    log_regression_knn(args.make_dataset)