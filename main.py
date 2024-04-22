import pandas as pd
import torch
import pickle

from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from model import ResidualNet
from make_dataset import make_dataset
from train import *
from dataset import *
from make_dataset import inverse_scaler, apply_scaler

def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def main():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Current CUDA device index: {device}")

    for i in range(10):
        train_dataset1, train_dataset2, test_dataset1, test_dataset2 = make_dataset()
        train_dataset1_name = './dataset/train_dataset1-' + str(i) + '.pkl'
        train_dataset2_name = './dataset/train_dataset2-' + str(i) + '.pkl'
        with open(train_dataset1_name, 'wb') as f:
            pickle.dump(train_dataset1, f)
        with open(train_dataset2_name, 'wb') as f:
            pickle.dump(train_dataset2, f)
        
        test_dataset1_name = './dataset/test_dataset1-' + str(i) + '.pkl'
        test_dataset2_name = './dataset/test_dataset2-' + str(i) + '.pkl'
        with open(test_dataset1_name, 'wb') as f:
            pickle.dump(test_dataset1, f)
        with open(test_dataset2_name, 'wb') as f:
            pickle.dump(test_dataset2, f)

        batch_size = 8
        train_dataset1, val_dataset1 = split_dataset(train_dataset1, seed=28, train_size=0.8)
        train_dataset2, val_dataset2 = split_dataset(train_dataset2, seed=28, train_size=0.8)

        train_loader1 = DataLoader(train_dataset1, batch_size, shuffle=True)
        val_loader1 = DataLoader(val_dataset1, batch_size, shuffle=False)
        train_loader2 = DataLoader(train_dataset2, batch_size, shuffle=True)
        val_loader2 = DataLoader(val_dataset2, batch_size, shuffle=False)
        print('dataloader load done')
        
        ##### model1: cured_height 예측 모델 / model2: PI, Do 예측 모델 #####
        model1 = ResidualNet(12, 4, 64, 1)
        model1.to(device)
        model1.apply(xavier_init)
        model2 = ResidualNet(12, 3, 64, 2)
        model2.to(device)
        model2.apply(xavier_init)

        print('model load done')

        ##### Model1 Training #####
        optimizer_ft = optim.Adam(model1.parameters(), lr=0.001)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)
        criterion = nn.MSELoss()
        model1, train_loss_list1, val_loss_list1 = train(model1, train_loader1, val_loader1, criterion, exp_lr_scheduler, optimizer_ft, 200, batch_size, device, 20)

        model1_name = 'model1-' + str(i)
        show_loss(train_loss_list1, val_loss_list1, model1_name + ' MSE Loss')
        model1.eval()
        torch.save(model1.state_dict(), './trained model/' + model1_name + '.pt')
        
        ##### Model2 Training #####
        optimizer_ft = optim.Adam(model2.parameters(), lr=0.001)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)
        criterion = nn.MSELoss()
        model2, train_loss_list2, val_loss_list2 = train(model2, train_loader2, val_loader2, criterion, exp_lr_scheduler, optimizer_ft, 200, batch_size, device, 20)

        model2_name = 'model2-' + str(i)
        show_loss(train_loss_list2, val_loss_list2, model2_name + ' MSE Loss')
        model2.eval()
        torch.save(model2.state_dict(), './trained model/' + model2_name + '.pt')

if __name__ == '__main__':
    main()