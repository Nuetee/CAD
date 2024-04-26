import pandas as pd
import torch
import pickle

from torch import nn, optim
from model import ResidualNet
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from dataset import *
from make_dataset import *

temporal_best_prediction_list = []
mean_prediction_list = []
intermediate_prediction_list = []
optimized_prediction_list = []
target_value_list = []

model1 = ResidualNet(8, 3, 64, 1)
model1.load_state_dict(torch.load('./trained model/' + 'model_total1.pt'))

model2 = ResidualNet(8, 2, 64, 2)
model2.load_state_dict(torch.load('./trained model/' + 'model_total2.pt'))

df_experiment = pd.read_csv('./data_experiment_2.csv')
# 'intensity_exposure', 'cured_height', test_dataset2 대용
test_data_list = []
for idx, row in df_experiment.iterrows():
    tensor = torch.tensor(row.values, dtype=torch.float32).unsqueeze(0)
    test_data_list.append(tensor)

model1.eval()
model2.eval()

loss_fn = torch.nn.MSELoss()
predictions = []

for test_data in test_data_list:
    outputs = model2(test_data)
    predictions.append(outputs)
    outputs_numpy = outputs.detach().numpy()
    
losses_and_predictions = []
with torch.no_grad():
    for prediction in predictions:
        mean_loss = 0
        count = 0

        for test_data in test_data_list:
            int_exp = test_data[:, :1]
            target = test_data[:, 1:]
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
for prediction in top_three_predictions:
    intermediate_prediction = prediction.clone().detach().requires_grad_()

    optimizer = optim.Adam([intermediate_prediction], lr=0.001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    optimized_prediction = 0
    attempt = 0
    min_loss = float('inf')
    while attempt < 3:
        mean_loss = 0
        for test_data in test_data_list:
            int_exp = test_data[:, :1]
            predicted_input = torch.cat((intermediate_prediction, int_exp), dim=1)
            outputs = model1(predicted_input)
            loss = loss_fn(outputs, target)
            mean_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        exp_lr_scheduler.step()
        mean_loss /= len(test_data_list)
        if min_loss > mean_loss:
            min_loss = mean_loss
            attempt = 0
        else:
            attempt += 1

    optimized_prediction = predicted_input[:, :2].clone()
    losses_and_predictions.append((loss, optimized_prediction))
top_predictions = sorted(losses_and_predictions, key=lambda x: x[0])[:1]
top_predictions = top_predictions[0][1]

intermediate_result = top_three_predictions[0].detach().numpy()
intermediate_result = intermediate_result[0]
top_predictions = top_predictions.detach().numpy()
top_predictions = top_predictions[0]
print(intermediate_result)
print(top_predictions)

with open('./scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

scaler = scalers['PI']
PI_inter = scaler.inverse_transform([[intermediate_result[0]]])
PI_opt = scaler.inverse_transform([[top_predictions[0]]])

scaler = scalers['Do']
Do_inter = scaler.inverse_transform([[intermediate_result[1]]])
Do_opt = scaler.inverse_transform([[top_predictions[1]]])

print(PI_inter, Do_inter)
print(PI_opt, Do_opt)