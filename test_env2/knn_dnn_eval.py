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
    df_origin = pd.read_csv('./data_minmax_scale.csv')
    mean_loss = 0
    mean_optimized_loss = 0
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Current CUDA device index: {device}")

    for i in range(10):
        with open('./dataset/train_dataset2-' + str(i) + '.pkl', 'rb') as f:
            train_dataset2 = pickle.load(f)
        with open('./dataset/test_dataset1-' + str(i) + '.pkl', 'rb') as f:
            test_dataset1 = pickle.load(f)
        with open('./dataset/test_dataset2-' + str(i) + '.pkl', 'rb') as f:
            test_dataset2 = pickle.load(f)

        model1 = ResidualNet(8, 4, 64, 1)
        model1_name = './trained model/model1-' + str(i) + '.pt'
        model1.load_state_dict(torch.load(model1_name))

        nearest = NearestNeighbors(n_neighbors=3)
        nearest.fit(train_dataset2.input)
        
        test_loader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False)
        test_loader2 =  DataLoader(test_dataset2, batch_size=len(test_dataset2), shuffle=False)

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
        
        intensity_repeated = np.repeat(intensity_values[0], len(x_data_filtered))
        new_inputs = np.column_stack((intensity_repeated, x_data_filtered, y_pred_filtered))
        
        if np.isnan(new_inputs).any():
            new_inputs = new_inputs[~np.isnan(new_inputs).any(axis=1)]
        new_inputs_filtered = new_inputs[new_inputs[:, 2] >= 0]
        distances, index = nearest.kneighbors(new_inputs_filtered)
        
        flattened_index = index.flatten()
        
        candidates = train_dataset2.target[flattened_index].view(-1, 2)
        candidates_rounded = np.round(candidates, decimals=4)

        candidates_list = candidates_rounded.numpy().tolist()
        counter = Counter(tuple(row) for row in candidates_list)
        most_common_candidate = counter.most_common(1)
        
        most_common_row, frequency = most_common_candidate[0]

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(torch.tensor(most_common_row), torch.tensor(target_data[0]))
        mean_loss += loss.item()

        most_common_tensor = torch.tensor(most_common_row)
        most_common_tensor = most_common_tensor.unsqueeze(0)
        optimizer = optim.Adam([most_common_tensor], lr=0.001)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        most_common_tensor.requires_grad_()
        model1.eval()

        attempt = 0
        min_loss = float('inf')
        while attempt < 3:
            mean_loss2 = 0
            for input, target in test_loader1:
                int_exp = input[:, 2:]
                predicted_input = torch.cat((most_common_tensor, int_exp), dim=1)
                outputs = model1(predicted_input)
                loss = loss_fn(outputs, target)
                mean_loss2 += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            exp_lr_scheduler.step()
            mean_loss2 /= len(test_loader1)
            if min_loss > mean_loss2:
                min_loss = mean_loss2
                attempt = 0
            else:
                attempt += 1
        
        optimized_prediction = predicted_input[:, :2].clone().detach()
        optimized_loss = loss_fn(optimized_prediction[0], torch.tensor(target_data[0]))
        print('iter' + str(i))
        print(most_common_tensor.detach(), target_data[0])
        mean_optimized_loss += optimized_loss.item()

    mean_loss /= 10
    mean_optimized_loss /= 10
    print('KNN mean loss: %f' % mean_loss)
    print('KNN-DNN mean loss: %f' % mean_optimized_loss)
if __name__ == "__main__":
    log_regression_with_DNN()