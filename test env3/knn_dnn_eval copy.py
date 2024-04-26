import pandas as pd
import torch
import pickle
import numpy as np

from collections import Counter
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from make_dataset import *
from dataset import *
from model import ResidualNet
from scipy.optimize import curve_fit
from train import *

def log_regression_with_DNN():
    df_scaled = pd.read_csv('./data_minmax_scale_2.csv')
    df_experiment = pd.read_csv('./data_experiment_2.csv')
    
    train_df_inputs = df_scaled[['intensity_exposure', 'cured_height']]
    train_df_targets = df_scaled[['PI', 'Do']]
    train_dataset = MyDataset(train_df_inputs, train_df_targets)
    nearest = NearestNeighbors(n_neighbors=3)
    nearest.fit(train_dataset.input)

    experiment_scaled_df = pd.read_csv('./data_experiment_2.csv')
    x_data = experiment_scaled_df['intensity_exposure'].to_numpy()
    y_data = experiment_scaled_df['cured_height'].to_numpy()
    
    # 모델 함수 정의
    def model_function(x, a, b, c, d):
        return np.log(b * x + c) / np.log(a) + d    
    
    # 초기 파라미터 추측
    initial_guess = [2, 1, 1, 1]
    params, params_covariance = curve_fit(model_function, x_data, y_data, p0=initial_guess)

    unique_exposure_times = df_scaled['intensity_exposure'].unique()
    x_data_all = np.sort(unique_exposure_times)

    mask = np.isin(x_data_all, x_data)
    close_mask = np.array([np.any(np.isclose(x, x_data)) for x in x_data_all])
    combined_mask = mask | close_mask

    # 근접한 값이나 정확히 일치하는 값을 제외합니다.
    x_data_filtered = x_data_all[~combined_mask]

    # x_data_to_pred_filtered에서 예측을 수행합니다.
    y_pred_filtered = model_function(x_data_filtered, *params)
    
    new_inputs = np.column_stack((x_data_filtered, y_pred_filtered))
    
    if np.isnan(new_inputs).any():
        new_inputs = new_inputs[~np.isnan(new_inputs).any(axis=1)]
    new_inputs_filtered = new_inputs[new_inputs[:, 1] >= 0]
    
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

    model1 = ResidualNet(8, 3, 64, 1)
    model1.load_state_dict(torch.load('./trained model/' + 'model_total1.pt'))
    
    most_common_tensor = torch.tensor(most_common_row)
    most_common_tensor = most_common_tensor.unsqueeze(0)
    optimizer = optim.Adam([most_common_tensor], lr=0.001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    most_common_tensor.requires_grad_()
    model1.eval()

    loss_fn = torch.nn.MSELoss()
    
    for idx, row in df_experiment.iterrows():
        tensor = torch.tensor(row.values, dtype=torch.float32).unsqueeze(0)
        int_exp = tensor[:, :1]
        target = tensor[:, 1:]

    attempt = 0
    min_loss = float('inf')
    print(most_common_tensor)
    while attempt < 3:
        predicted_input = torch.cat((most_common_tensor, int_exp), dim=1)
        outputs = model1(predicted_input)
        loss = loss_fn(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        exp_lr_scheduler.step()
        if min_loss > loss.item():
            min_loss = loss.item()
            attempt = 0
        else:
            attempt += 1
    
    optimized_prediction = predicted_input[:, :2].clone().detach()
    print(optimized_prediction)
    optimized_prediction = optimized_prediction.detach().numpy()
    optimized_prediction = optimized_prediction[0]

    with open('./scalers_2.pkl', 'rb') as f:
        scalers = pickle.load(f)

    scaler = scalers['PI']
    PI = scaler.inverse_transform([[optimized_prediction[0]]])

    scaler = scalers['Do']
    Do = scaler.inverse_transform([[optimized_prediction[1]]])

    print(PI, Do)
if __name__ == "__main__":
    log_regression_with_DNN()