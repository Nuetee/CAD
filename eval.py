import pandas as pd
import torch
import pickle

from torch import nn, optim
from model import ResidualNet
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from dataset import *

temporal_best_prediction_list = []
mean_prediction_list = []
intermediate_prediction_list = []
optimized_prediction_list = []
target_value_list = []

for i in range(10):
    # model1 = ResidualNet(12, 4, 64, 1)
    # model1.load_state_dict(torch.load('./trained model/' + 'model1-' + str(i) +'.pt'))

    # model2 = ResidualNet(12, 3, 64, 2)
    # model2.load_state_dict(torch.load('./trained model/' + 'model2-' + str(i) +'.pt'))

    ##### KNN ######
    model1 = ResidualNet(12, 4, 64, 1)
    model1.load_state_dict(torch.load('./trained model2/model1_layer12.pt'))

    with open('./dataset2/train_dataset2.pkl', 'rb') as f:
        train_dataset2 = pickle.load(f)
    with open('./dataset2/test_dataset1-' + str(i) + '.pkl', 'rb') as f:
        test_dataset1 = pickle.load(f)
    with open('./dataset2/test_dataset2-' + str(i) + '.pkl', 'rb') as f:
        test_dataset2 = pickle.load(f)
    
    nearest = NearestNeighbors(n_neighbors=1)
    nearest.fit(train_dataset2.input)
    ##### KNN ######

    # with open('./dataset/test_dataset1-' + str(i) + '.pkl', 'rb') as f:
    #     test_dataset1 = pickle.load(f)
    # with open('./dataset/test_dataset2-' + str(i) + '.pkl', 'rb') as f:
    #     test_dataset2 = pickle.load(f)

    test_loader1 = DataLoader(test_dataset1, batch_size=1, shuffle=False)
    test_loader2 = DataLoader(test_dataset2, batch_size=1, shuffle=False)

    model1.eval()
    # model2.eval()

    loss_fn = torch.nn.MSELoss()
    predictions = []

    temp_min_loss = float('inf')
    
    ##### KNN ##### 
    knn_list = []
    best_prediction = 0
    for inputs, target in test_loader2:
        distances, index = nearest.kneighbors(inputs)
        knn_list.append(train_dataset2.target[index])
        loss = loss_fn(target, train_dataset2.target[index])
        if loss < temp_min_loss:
            temp_min_loss = loss
            best_prediction = train_dataset2.target[index]        
    ##### KNN #####

    # with torch.no_grad():
    #     best_prediction = 0
    #     for inputs, target in test_loader2:
    #         outputs = model2(inputs)
    #         loss = loss_fn(target, outputs)
    #         if loss < temp_min_loss:
    #             temp_min_loss = loss
    #             best_prediction = outputs
    #         predictions.append(outputs)  # 차원 추가
        
        temporal_best_prediction_list.append(best_prediction)
    
    # mean_prediction = sum(predictions) / len(predictions)
    # mean_prediction_list.append(mean_prediction)
    
    ##### KNN #####
    mean_prediction = sum(knn_list) / len(knn_list)
    mean_prediction_list.append(mean_prediction)
    ##### KNN #####

    min_loss = float('inf')
    
    target_value = 0
    with torch.no_grad():
        for _, target in test_loader2:
            target_value = target
            break
    
    target_value_list.append(target_value)

    losses_and_predictions = []
    with torch.no_grad():
        for prediction in predictions:
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
            optimized_prediction = predicted_input[:, :2].clone()

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

print(target_value_list)
print(mean_prediction_list)
print(temporal_best_prediction_list)
print(intermediate_prediction_list)
print(optimized_prediction_list)
# with open('targets.pkl', 'wb') as f:
#     pickle.dump(target_value_list, f)
# with open('mean_predictions.pkl', 'wb') as f:
#     pickle.dump(mean_prediction_list, f)
# with open('nearest_predictions.pkl', 'wb') as f:
#     pickle.dump(temporal_best_prediction_list, f)
# with open('top_three_intermediate_predictions.pkl', 'wb') as f:
#     pickle.dump(intermediate_prediction_list, f)
# with open('top_three_optimized_predictions.pkl', 'wb') as f:
#     pickle.dump(optimized_prediction_list, f)