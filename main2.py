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
    parser = argparse.ArgumentParser(description="A simple file processing script.")
    parser.add_argument('--make-dataset', type=int, default=0)
    parser.add_argument('--l1', type=float, default=0)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--model1-name', type=str, default='model1_layer')
    parser.add_argument('--model2-name', type=str, default='model2_layer')
    
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Current CUDA device index: {device}")
    
    if args.make_dataset != 0:
        train_dataset1, train_dataset2, test_dataset1_list, test_dataset2_list = make_dataset2()
        train_dataset1_name = './dataset2/train_dataset1.pkl'
        train_dataset2_name = './dataset2/train_dataset2.pkl'
        
        with open(train_dataset1_name, 'wb') as f:
            pickle.dump(train_dataset1, f)
        with open(train_dataset2_name, 'wb') as f:
            pickle.dump(train_dataset2, f)
        
        for i in range(10):
            test_dataset1_name = './dataset2/test_dataset1-' + str(i) + '.pkl'
            test_dataset2_name = './dataset2/test_dataset2-' + str(i) + '.pkl'
            
            with open(test_dataset1_name, 'wb') as f:
                pickle.dump(test_dataset1_list[i], f)
            with open(test_dataset2_name, 'wb') as f:
                pickle.dump(test_dataset2_list[i], f)
    else:
        with open('./dataset2/train_dataset1.pkl', 'rb') as f:
            train_dataset1 = pickle.load(f)
        with open('./dataset2/train_dataset2.pkl', 'rb') as f:
            train_dataset2 = pickle.load(f)

    batch_size = 8
    train_loader1 = DataLoader(train_dataset1, batch_size, shuffle=True)
    train_loader2 = DataLoader(train_dataset2, batch_size, shuffle=True)
    print('dataloader load done')

    for layer_num in [4, 8, 16]:
        model1 = ResidualNet(layer_num, 4, 64, 1)
        model1.to(device)
        model1.apply(xavier_init)
        model2 = ResidualNet(layer_num, 3, 64, 2)
        model2.to(device)
        model2.apply(xavier_init)

        weight_decay = args.l2
        optimizer_ft = optim.Adam(model1.parameters(), lr=0.001, weight_decay=weight_decay)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)
        criterion = nn.MSELoss()
        model1, _, _ = train(model1, train_loader1, criterion, exp_lr_scheduler, optimizer_ft, 50, batch_size, device, val_dataloader=None, patience=20, is_val=False, lambda1=args.l1)

        torch.save(model1.state_dict(), './trained model2/' + args.model1_name + str(layer_num) + '.pt')

        ##### Model2 Training #####
        optimizer_ft = optim.Adam(model2.parameters(), lr=0.001, weight_decay=weight_decay)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)
        criterion = nn.MSELoss()
        model2, _, _ = train(model2, train_loader2, criterion, exp_lr_scheduler, optimizer_ft, 50, batch_size, device, val_dataloader=None, patience=20, is_val=False, lambda1=args.l1)

        torch.save(model2.state_dict(), './trained model2/' + args.model2_name + str(layer_num) + '.pt')

if __name__ == '__main__':
    main()