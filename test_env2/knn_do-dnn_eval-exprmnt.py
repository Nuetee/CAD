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

def log_regression_with_Do_DNN():
    df_scaled = pd.read_csv('./data_minmax_scale.csv')
    df_experiment = pd.read_csv('./data_experiment.csv')
    
    train_df_inputs = df_scaled[['intensity', 'exposure_time', 'cured_height']]
    train_df_targets = df_scaled[['PI', 'Do']]
    train_dataset = MyDataset(train_df_inputs, train_df_targets)
    nearest = NearestNeighbors(n_neighbors=3)
    nearest.fit(train_dataset.input)

    experiment_scaled_df = pd.read_csv('./data_experiment.csv')
    grouped = experiment_scaled_df.groupby('intensity')

    for name, group in grouped:
        x_data = group['exposure_time'].to_numpy()
        y_data = group['cured_height'].to_numpy()
        if len(x_data) < 4:
            def model_function(x, params):
                a, b, c, d = params[0], params[1], params[2], params[3]
                return torch.log(b * x + c) / torch.log(a) + d

            x = torch.tensor(group['exposure_time'].values, dtype=torch.float32)
            y = torch.tensor(group['cured_height'].values, dtype=torch.float32)
            
            params = torch.tensor([2.0, 1.0, 1.0, 0.0], requires_grad=True)
            criterion = nn.MSELoss()
            optimizer = optim.Adam([params], lr=0.001)
            min_loss =float('inf')
            attempt = 0
            while attempt < 10:
                optimizer.zero_grad()
                output = model_function(x, params)
                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss.item() < min_loss:
                    min_loss = loss.item()
                    attempt = 0
                else:
                    attempt += 1
        else:
            # 모델 함수 정의
            def model_function(x, a, b, c, d):
                return np.log(b * x + c) / np.log(a) + d    
            
            # 초기 파라미터 추측name
            initial_guess = [2, 1, 1, 1]
            params, params_covariance = curve_fit(model_function, x_data, y_data, p0=initial_guess)

        unique_exposure_times = df_scaled['exposure_time'].unique()
        x_data_all = np.sort(unique_exposure_times)

        mask = np.isin(x_data_all, x_data)
        close_mask = np.array([np.any(np.isclose(x, x_data)) for x in x_data_all])
        combined_mask = mask | close_mask

        # 근접한 값이나 정확히 일치하는 값을 제외합니다.
        x_data_filtered = x_data_all[~combined_mask]

        # x_data_to_pred_filtered에서 예측을 수행합니다.
        if len(x_data) < 4:
            x_data_to_pred_filtered_tensor = torch.tensor(x_data_filtered)
            y_pred_filtered = model_function(x_data_to_pred_filtered_tensor, params).detach().numpy()
        else:
            y_pred_filtered = model_function(x_data_filtered, *params)
        
        intensity_repeated = np.repeat(name, len(x_data_filtered))
        new_inputs = np.column_stack((intensity_repeated, x_data_filtered, y_pred_filtered))
        
        if np.isnan(new_inputs).any():
            new_inputs = new_inputs[~np.isnan(new_inputs).any(axis=1)]
        new_inputs_filtered = new_inputs[new_inputs[:, 2] >= 0]
        distances, index = nearest.kneighbors(new_inputs_filtered)
        
        flattened_index = index.flatten()
        
        candidates = train_dataset.target[flattened_index].view(-1, 2)
        candidates_rounded = np.round(candidates, decimals=4)

        candidates_list = candidates_rounded.numpy().tolist()
        counter = Counter(tuple(row) for row in candidates_list)
        most_common_candidate = counter.most_common(1)
        
        most_common_row, frequency = most_common_candidate[0]

        model3 = ResidualNet(8, 4, 64, 1)
        model3_name = './trained model/model3-1.pt'
        model3.load_state_dict(torch.load(model3_name))
        model3.eval()

        loss_fn = torch.nn.MSELoss()

        pi = torch.tensor([most_common_row[0]]).unsqueeze(0)
        pi.requires_grad_()
        
        row_tensor_list = []
        for idx, row in group.iterrows():
            tensor = torch.tensor(row.values, dtype=torch.float32).unsqueeze(0)
            row_tensor_list.append(tensor)
        
        ###### model3로 Do 후보들 생성 #####
        pi_do_candidates = []
        
        with torch.no_grad():
            for row_tensor in row_tensor_list:
                int_exp_ch = row_tensor[:, :3]
                predicted_input = torch.cat((int_exp_ch, pi), dim=1)
                do = model3(predicted_input)
                pi_do_candidates.append(torch.cat((pi, do), dim=1))

        ###### model1으로 Do 후보들중 top3 선정 #####
        
        model1 = ResidualNet(8, 4, 64, 1)
        model1.load_state_dict(torch.load('./trained model/' + 'model_total1_l2.pt'))
        model1.eval()

        losses_and_predictions = []
        with torch.no_grad():
            for prediction in pi_do_candidates:
                mean_loss = 0
                count = 0
                for row_tensor in row_tensor_list:
                    int_exp = row_tensor[:, :2]
                    target = row_tensor[:, 2:]
                    predicted_input = torch.cat((prediction, int_exp), dim=1)
                    outputs = model1(predicted_input)
                    
                    loss = loss_fn(target, outputs)
                    mean_loss += loss.item()
                    count += 1
                
                mean_loss /= count
                losses_and_predictions.append((mean_loss, prediction))
        
        top_three_predictions = sorted(losses_and_predictions, key=lambda x: x[0])[:3]
        top_three_predictions = [prediction[1] for prediction in top_three_predictions]
        
        losses_and_predictions = []
        ##### model1으로 최적 Do 생성 ######
        for prediction in top_three_predictions:
            pi = prediction[:, 0:1].clone()
            do = prediction[:, 1:2].clone().detach().requires_grad_()
            
            optimizer = optim.Adam([do], lr=0.001)
            exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

            optimized_prediction = 0
            attempt = 0
            min_loss = float('inf')
            while attempt < 3:
                mean_loss = 0
                for row_tensor in row_tensor_list:
                    int_exp = row_tensor[:, :2]
                    predicted_input = torch.cat((pi, do, int_exp), dim=1)
                    outputs = model1(predicted_input)
                    loss = loss_fn(outputs, target)
                    mean_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                exp_lr_scheduler.step()
                mean_loss /= len(row_tensor_list)
                if min_loss > mean_loss:
                    min_loss = mean_loss
                    attempt = 0
                else:
                    attempt += 1

            optimized_prediction = predicted_input[:, :2].clone()
            losses_and_predictions.append((loss.item(), optimized_prediction))
        top_predictions = sorted(losses_and_predictions, key=lambda x: x[0])[:1]
        top_prediction = top_predictions[0][1][0].detach().numpy()

        with open('./scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)

        scaler = scalers['PI']
        pi = scaler.inverse_transform([[top_prediction[0]]])
        scaler = scalers['Do']
        Do = scaler.inverse_transform([[top_prediction[1]]])
        # print(pi[0][0], Do[0][0])
        
        print(x_data_filtered[-3:])
        print(y_pred_filtered[-3:])
        # most_common_index = candidates_list.index(list(most_common_row))
        # print(most_common_row)
        for i in range(3):
            scaler = scalers['exposure_time']
            pi = scaler.inverse_transform([[x_data_filtered[-3:][i]]])
            scaler = scalers['cured_height']
            Do = scaler.inverse_transform([[y_pred_filtered[-3:][i]]])
            print(Do[0][0])
        # cd2 = train_dataset.input[flattened_index].view(-1, 2)
        for candidate in candidates_list:
            scaler = scalers['PI']
            pi = scaler.inverse_transform([[candidate[0]]])
            scaler = scalers['Do']
            Do = scaler.inverse_transform([[candidate[1]]])
            # print(Do[0][0])
        
        break
    

if __name__ == '__main__':
    log_regression_with_Do_DNN()