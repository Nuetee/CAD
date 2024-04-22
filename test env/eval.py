import pandas as pd
import torch
import pickle

from torch import nn, optim
from model import ResidualNet
from torch.utils.data import DataLoader

mean_prediction_list = []
intermediate_prediction_list = []
optimized_prediction_list = []
target_value_list = []

for i in range(10):
    model1 = ResidualNet(12, 4, 64, 1)
    model1.load_state_dict(torch.load('./trained model/' + 'model1-' + str(i) +'.pt'))

    model2 = ResidualNet(12, 3, 64, 2)
    model2.load_state_dict(torch.load('./trained model/' + 'model2-' + str(i) +'.pt'))

    with open('./dataset/test_dataset1-' + str(i) + '.pkl', 'rb') as f:
        test_dataset1 = pickle.load(f)
    with open('./dataset/test_dataset2-' + str(i) + '.pkl', 'rb') as f:
        test_dataset2 = pickle.load(f)
    
    test_loader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False)
    test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False)

    model1.eval()
    model2.eval()

    predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader2:
            outputs = model2(inputs)
            predictions.append(outputs)  # 차원 추가
    
    mean_prediction = sum(predictions) / len(predictions)
    mean_prediction_list.append(mean_prediction)

    loss_fn = torch.nn.MSELoss()
    min_loss = float('inf')
    
    target_value = 0
    with torch.no_grad():
        for _, target in test_loader2:
            target_value = target
            break
    
    target_value_list.append(target_value)

    with torch.no_grad():
        intermediate_prediction = 0
        for prediction in predictions:
            mean_loss = 0
            for i, (inputs, target) in enumerate(test_loader1):
                int_exp = inputs[:, 2:]
                predicted_input = torch.cat((prediction, int_exp), dim=1)
                outputs = model1(predicted_input)
                
                loss = loss_fn(target, outputs)
                mean_loss += loss
            
            mean_loss /= len(test_loader1)
            if min_loss > mean_loss:
                min_loss = mean_loss
                intermediate_prediction = prediction
    
    intermediate_prediction_list.append(intermediate_prediction.clone())

    optimizer = optim.Adam([intermediate_prediction], lr=0.001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    intermediate_prediction.requires_grad_()

    optimized_prediction = 0
    for epoch in range(10):
        for input, target in test_loader1:
            predicted_input = torch.cat((intermediate_prediction, int_exp), dim=1)
            outputs = model1(predicted_input)
            loss = loss_fn(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        exp_lr_scheduler.step()
        optimized_prediction = predicted_input[:, :2].clone()

    optimized_prediction_list.append(optimized_prediction)

loss_fn = torch.nn.MSELoss()
mean_prediction_mean_loss = 0
intermediate_prediction_mean_loss = 0
optimized_prediction_mean_loss = 0

for i, target_value in enumerate(target_value_list):
    mean_prediction_mean_loss += loss_fn(target_value, mean_prediction_list[i])
    intermediate_prediction_mean_loss += loss_fn(target_value, intermediate_prediction_list[i])
    optimized_prediction_mean_loss += loss_fn(target_value, optimized_prediction_list[i])

    print("(%f, %f) : (%f, %f) (%f, %f) (%f, %f)" % (target_value[0][0].item(), target_value[0][1].item(), mean_prediction_list[i][0][0].item(), mean_prediction_list[i][0][1].item(), intermediate_prediction_list[i][0][0].item(), intermediate_prediction_list[i][0][1].item(),optimized_prediction_list[i][0][0].item(), optimized_prediction_list[i][0][1].item()))

mean_prediction_mean_loss /= len(target_value_list)
intermediate_prediction_mean_loss /= len(target_value_list)
optimized_prediction_mean_loss /= len(target_value_list)

print('Mean predictions loss(mean): %f' % mean_prediction_mean_loss)
print('Intermediate predictions loss(mean): %f' % intermediate_prediction_mean_loss)
print('Optimized predictions loss(mean): %f' % optimized_prediction_mean_loss)