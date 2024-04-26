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

def knn():
    with open('./dataset2/train_dataset2.pkl', 'rb') as f:
        train_dataset2 = pickle.load(f)
    nearest = NearestNeighbors(n_neighbors=1)
    nearest.fit(train_dataset2.input)

    temporal_best_prediction_list = []
    mean_prediction_list = []
    intermediate_prediction_list = []
    optimized_prediction_list = []
    target_value_list = []

    for i in range(10):
        model1 = ResidualNet(6, 4, 64, 1)
        model1.load_state_dict(torch.load('./trained model2/' + 'model1.pt'))

        with open('./dataset2/test_dataset1-' + str(i) + '.pkl', 'rb') as f:
            test_dataset1 = pickle.load(f)
        with open('./dataset2/test_dataset2-' + str(i) + '.pkl', 'rb') as f:
            test_dataset2 = pickle.load(f)

        test_loader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False)
        test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False)

        loss_fn = torch.nn.MSELoss()
        temp_min_loss = float('inf')
        knn_list = []
        best_prediction = 0
        for inputs, target in test_loader2:
            distances, index = nearest.kneighbors(inputs)
            knn_list.append(train_dataset2.target[index])
            loss = loss_fn(target, train_dataset2.target[index])
            if loss < temp_min_loss:
                temp_min_loss = loss
                best_prediction = train_dataset2.target[index]
                
        temporal_best_prediction_list.append(best_prediction)

        mean_prediction = sum(knn_list) / len(knn_list)
        mean_prediction_list.append(mean_prediction)

        min_loss = float('inf')
            
        target_value = 0
        with torch.no_grad():
            for _, target in test_loader2:
                target_value = target
                break
        target_value_list.append(target_value)
        
        model1.eval()
        losses_and_predictions = []
        with torch.no_grad():
            for nearest_neighbor in knn_list:
                mean_loss = 0
                count = 0
                for i, (inputs, target) in enumerate(test_loader1):
                    int_exp = inputs[:, 2:]
                    predicted_input = torch.cat((nearest_neighbor, int_exp), dim=1)
                    outputs = model1(predicted_input)
                    
                    loss = loss_fn(target, outputs)
                    mean_loss += loss.item()
                    count += 1
                
                mean_loss /= count
                losses_and_predictions.append((mean_loss, nearest_neighbor))

        top_three_predictions = sorted(losses_and_predictions, key=lambda x: x[0])[:3]
        top_three_predictions = [prediction[1] for prediction in top_three_predictions]

        intermediate_prediction_list.append(top_three_predictions)

        top_three_optimized_predictions = []
        for prediction in top_three_predictions:
            intermediate_prediction = prediction.clone()

            optimizer = optim.Adam([intermediate_prediction], lr=0.001)
            exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            intermediate_prediction.requires_grad_()

            optimized_prediction = 0
            for epoch in range(5):
                for input, target in test_loader1:
                    predicted_input = torch.cat((intermediate_prediction, int_exp), dim=1)
                    outputs = model1(predicted_input)
                    loss = loss_fn(outputs, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                exp_lr_scheduler.step()
                optimized_prediction = predicted_input[:, :2].clone().detach()

            top_three_optimized_predictions.append(optimized_prediction)
        optimized_prediction_list.append(top_three_optimized_predictions)

    loss_fn = torch.nn.MSELoss()
    temporal_best_prediction_mean_loss = 0
    mean_prediction_mean_loss = 0
    intermediate_prediction_mean_loss = [0, 0, 0]
    optimized_prediction_mean_loss = [0, 0, 0]

    for i, target_value in enumerate(target_value_list):
        print("Test data %d에 대한 Prediction" % (i))
        print("Target (PI, Do): (%f, %f)" % (target_value[0][0].item(), target_value[0][1].item()))
        print("The Nearest prediction with target (PI, Do) : (%f, %f)  /  Loss: %f" % (temporal_best_prediction_list[i][0][0].item(), temporal_best_prediction_list[i][0][1].item(), loss_fn(target_value, temporal_best_prediction_list[i]).item()))
        print("Mean prediction with target (PI, Do) : (%f, %f)  /  Loss: %f" % (mean_prediction_list[i][0][0].item(), mean_prediction_list[i][0][1].item(), loss_fn(target_value, mean_prediction_list[i]).item()))
        for j in range(3):
            print("%d번째 prediction (PI, Do): (%f, %f)  /  Loss: %f" % (j, intermediate_prediction_list[i][j][0][0].item(), intermediate_prediction_list[i][j][0][0].item(), loss_fn(target_value, intermediate_prediction_list[i][j]).item()))
            print("%d번째 optimized prediction (PI, Do): (%f, %f)  /  Loss: %f" % (j, optimized_prediction_list[i][j][0][0].item(), optimized_prediction_list[i][j][0][0].item(), loss_fn(target_value, optimized_prediction_list[i][j]).item()))
            print('\n')
        
        temporal_best_prediction_mean_loss += loss_fn(target_value, temporal_best_prediction_list[i])
        mean_prediction_mean_loss += loss_fn(target_value, mean_prediction_list[i])
        for j in range(3):
            intermediate_prediction_mean_loss[j] += loss_fn(target_value, intermediate_prediction_list[i][j])
            optimized_prediction_mean_loss[j] += loss_fn(target_value, optimized_prediction_list[i][j])

    temporal_best_prediction_mean_loss /= len(target_value_list)
    mean_prediction_mean_loss /= len(target_value_list)
    for intermediate_prediction in intermediate_prediction_mean_loss:
        intermediate_prediction /= len(target_value_list)
    for optimized_prediction in optimized_prediction_mean_loss:
        optimized_prediction /= len(target_value_list)

    print('Temporal best predictions loss(mean): %f' % temporal_best_prediction_mean_loss)
    print('Mean predictions loss(mean): %f' % mean_prediction_mean_loss)
    for i in range(3):
        print('%d번째 Intermediate predictions loss(mean): %f' % (i, intermediate_prediction_mean_loss[i]))
        print('%d번째 Optimized predictions loss(mean): %f' % (i, optimized_prediction_mean_loss[i]))

def log_regression_knn():
    df_origin = pd.read_csv('./data_minmax_scale.csv')
    mean_loss = 0

    for i in range(10):
        _, train_dataset, _, test_dataset = make_dataset3()
        nearest = NearestNeighbors(n_neighbors=3)
        nearest.fit(train_dataset.input)

        test_loader2 =  DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

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
        loss = loss_fn(torch.tensor(most_common_row), torch.tensor(target_data[0]))
        print("Loss: %f" % (loss))
        mean_loss += loss
    mean_loss /= 10
    print('Mean Loss: %f' % (mean_loss))

if __name__ == "__main__":
    log_regression_knn()