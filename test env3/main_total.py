import pandas as pd
import torch
import pickle
import argparse

from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from model import ResidualNet
from make_dataset import *
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

    df_scaled = pd.read_csv('./data_minmax_scale_2.csv')

    train_df_inputs1 = df_scaled[['PI', 'Do', 'intensity_exposure']]
    train_df_targets1 = df_scaled[['cured_height']]
    train_dataset1 = MyDataset(train_df_inputs1, train_df_targets1)
    
    train_df_inputs2 = df_scaled[['intensity_exposure', 'cured_height']]
    train_df_targets2 = df_scaled[['PI', 'Do']]
    train_dataset2 = MyDataset(train_df_inputs2, train_df_targets2)

    batch_size = 32

    train_loader1 = DataLoader(train_dataset1, batch_size, shuffle=True)
    print('dataloader load done')
    
    ##### model1: cured_height 예측 모델 #####
    model1 = ResidualNet(8, 3, 64, 1)
    model1.to(device)
    model1.apply(xavier_init)
    
    print('model1 load done')

    ##### Model1 Training #####
    optimizer_ft = optim.Adam(model1.parameters(), lr=0.001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)
    criterion = nn.MSELoss()
    model1, train_loss_list1 = train(model1, train_loader1, criterion, exp_lr_scheduler, optimizer_ft, 50, batch_size, device)

    model1_name = 'model_total1'
    model1.eval()
    torch.save(model1.state_dict(), './trained model/' + model1_name + '.pt')

    train_loader2 = DataLoader(train_dataset2, batch_size, shuffle=True)
    print('dataloader load done')
    
    ##### model2: PI, Do 예측 모델 #####
    model2 = ResidualNet(8, 2, 64, 2)
    model2.to(device)
    model2.apply(xavier_init)
    
    print('model2 load done')

    ##### Model2 Training #####
    optimizer_ft = optim.Adam(model2.parameters(), lr=0.001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)
    criterion = nn.MSELoss()
    model2, train_loss_list1 = train(model2, train_loader2, criterion, exp_lr_scheduler, optimizer_ft, 50, batch_size, device)

    model2_name = 'model_total2'
    model2.eval()
    torch.save(model2.state_dict(), './trained model/' + model2_name + '.pt')

if __name__ == '__main__':
    main()