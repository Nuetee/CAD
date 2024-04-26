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
    temporal_best_prediction_list = []
    intermediate_prediction_list = []
    optimized_prediction_list = []
    target_value_list = []

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
    
        model3 = ResidualNet(8, 4, 64, 1)
        model3_name = './trained model/model3-' + str(i) + '.pt'
        model3.load_state_dict(torch.load(model3_name))
        model3.eval()

        model1 = ResidualNet(8, 4, 64, 1)
        model1.load_state_dict(torch.load('./trained model/' + 'model1-' + str(i) +'.pt'))
        model1.eval()

        nearest = NearestNeighbors(n_neighbors=3)
        nearest.fit(train_dataset2.input)

        test_loader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False)
        test_loader2 =  DataLoader(test_dataset2, batch_size=len(test_dataset2), shuffle=False)
        
        target_value = 0
        with torch.no_grad():
            for _, target in test_loader2:
                target_value = target
                break
        
        target_value_list.append(target_value)
        
        ###### KNN으로 PI 값 추정 ######
        for inputs, targets in test_loader2:
            intensity_values = inputs[:, 0].numpy()  # 'intensity' 추정: 0번째 열
            x_data = inputs[:, 1].numpy()  # 'exposure_time' 추정: 1번째 열
            y_data = inputs[:, 2].numpy()  # 'cured_height' 추정: 2번째 열
            target_data = targets[:, ].numpy()
        
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

        pi = torch.tensor([most_common_row[0]])
        pi = pi.unsqueeze(0)

        ###### model3로 Do 후보들 생성 #####
        pi_do_candidates = []
        test_loader2_1 =  DataLoader(test_dataset2, batch_size=1, shuffle=False)
        temp_min_loss = float('inf')
        with torch.no_grad():
            best_prediction = 0
            for input, target in test_loader2_1:
                int_exp_ch = input
                predicted_input = torch.cat((int_exp_ch, pi), dim=1)
                do = model3(predicted_input)
                loss = loss_fn(target[:, 1:2], do)
                if loss < temp_min_loss:
                    temp_min_loss = loss
                    best_prediction = do
                pi_do_candidates.append(torch.cat((pi, do), dim=1))
            temporal_best_prediction_list.append(torch.cat((pi, best_prediction), dim=1))
            
        ###### model1으로 Do 후보들중 top3 선정 #####
        losses_and_predictions = []
        with torch.no_grad():
            for prediction in pi_do_candidates:
                mean_loss = 0
                count = 0
                for i, (inputs, target) in enumerate(test_loader1):
                    int_exp = inputs[:, 2:]
                    predicted_input = torch.cat((prediction, int_exp), dim=1)
                    outputs = model1(predicted_input)
                    
                    loss = loss_fn(target, outputs)
                    mean_loss += loss.item()
                    count += 1
                
                mean_loss /= count
                losses_and_predictions.append((mean_loss, prediction))
        
        top_three_predictions = sorted(losses_and_predictions, key=lambda x: x[0])[:3]
        top_three_predictions = [prediction[1] for prediction in top_three_predictions]
        intermediate_prediction_list.append(top_three_predictions)

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
                for input, target in test_loader1:
                    int_exp = input[:, 2:]
                    predicted_input = torch.cat((pi, do, int_exp), dim=1)
                    outputs = model1(predicted_input)
                    loss = loss_fn(outputs, target)
                    mean_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                exp_lr_scheduler.step()
                mean_loss /= len(test_loader1)
                if min_loss > mean_loss:
                    min_loss = mean_loss
                    attempt = 0
                else:
                    attempt += 1

            optimized_prediction = predicted_input[:, :2].clone()
            losses_and_predictions.append((loss.item(), optimized_prediction))
        top_predictions = sorted(losses_and_predictions, key=lambda x: x[0])[:1]
        top_predictions = top_predictions[0][1]

        optimized_prediction_list.append(top_predictions)
    loss_fn = torch.nn.MSELoss()
    temporal_best_prediction_mean_loss = 0
    intermediate_prediction_mean_loss = [0, 0, 0]
    optimized_prediction_mean_loss = 0

    for i, target_value in enumerate(target_value_list):
        print("Test data %d에 대한 Prediction" % (i))
        print("Target (PI, Do): (%f, %f)" % (target_value[0][0].item(), target_value[0][1].item()))
        print("The Nearest prediction with target (PI, Do) : (%f, %f)  /  Loss: %f" % (temporal_best_prediction_list[i][0][0].item(), temporal_best_prediction_list[i][0][1].item(), loss_fn(target_value, temporal_best_prediction_list[i]).item()))
        for j in range(3):
            print("%d번째 prediction (PI, Do): (%f, %f)  /  Loss: %f" % (j, intermediate_prediction_list[i][j][0][0].item(), intermediate_prediction_list[i][j][0][1].item(), loss_fn(target_value, intermediate_prediction_list[i][j]).item()))
        
        print("Optimized prediction (PI, Do): (%f, %f)  /  Loss: %f" % ( optimized_prediction_list[i][0][0].item(), optimized_prediction_list[i][0][1].item(), loss_fn(target_value, optimized_prediction_list[i]).item()))
        print('\n')
        
        temporal_best_prediction_mean_loss += loss_fn(target_value, temporal_best_prediction_list[i])
        optimized_prediction_mean_loss += loss_fn(target_value, optimized_prediction_list[i])
        for j in range(3):
            intermediate_prediction_mean_loss[j] += loss_fn(target_value, intermediate_prediction_list[i][j])
    
    temporal_best_prediction_mean_loss /= len(target_value_list)
    optimized_prediction_mean_loss /= len(target_value_list)
    for intermediate_prediction in intermediate_prediction_mean_loss:
        intermediate_prediction /= len(target_value_list)            
    
    print('Temporal best predictions loss(mean): %f' % temporal_best_prediction_mean_loss)
    for i in range(3):
        print('%d번째 Intermediate predictions loss(mean): %f' % (i, intermediate_prediction_mean_loss[i]))
    print('Optimized predictions loss(mean): %f' % (optimized_prediction_mean_loss))
        # optimized_prediction = predicted_input[:, :2].clone().detach()
        
        # optimized_loss = loss_fn(optimized_prediction[0], torch.tensor(target_data[0]))
        # print('iter' + str(i))
        # print(most_common_row, pi.detach(), target_data[0])
        # mean_optimized_loss += optimized_loss.item()
    
    # mean_loss /= 10
    # mean_optimized_loss /= 10
    # print('KNN mean loss: %f' % mean_loss)
    # print('KNN-DNN mean loss: %f' % mean_optimized_loss)

if __name__ == '__main__':
    log_regression_with_Do_DNN()