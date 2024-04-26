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
    
    for i in range(10):
        with open('./dataset/train_dataset3-' + str(i) + '.pkl', 'rb') as f:
            train_dataset3 = pickle.load(f)
            
        batch_size = 32
        train_loader3 = DataLoader(train_dataset3, batch_size, shuffle=True)
        print('dataloader load done')
        
        ##### model#: Do 예측 모델 #####
        model3 = ResidualNet(8, 4, 64, 1)
        model3.to(device)
        model3.apply(xavier_init)
        
        print('model3 load done')

        ##### Model3 Training #####
        optimizer_ft = optim.Adam(model3.parameters(), lr=0.001)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)
        criterion = nn.MSELoss()
        model3, train_loss_list1 = train(model3, train_loader3, criterion, exp_lr_scheduler, optimizer_ft, 50, batch_size, device)

        model3_name = 'model3-' + str(i)
        model3.eval()
        torch.save(model3.state_dict(), './trained model/' + model3_name + '.pt')

        # with open('./dataset/train_dataset3-' + str(i) + '.pkl', 'rb') as f:
        #     train_dataset2 = pickle.load(f)

        # new_input = torch.cat((train_dataset2.input, train_dataset2.target[:, 0:1]), dim=1)
        # new_target = train_dataset2.target[:, 1:2]

        # new_train_dataset = MyDataset(new_input, new_target)
        # test_dataset3_name = './dataset/test_dataset3-' + str(i) + '.pkl'
        # with open(test_dataset3_name, 'wb') as f:
        #     pickle.dump(new_train_dataset, f)
if __name__ ==  '__main__':
    main()