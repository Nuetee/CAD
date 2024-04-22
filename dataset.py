import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split

class MyDataset(Dataset):
    def __init__(self, input, target):
        self.input = torch.tensor(input.values, dtype=torch.float32)
        self.target = torch.tensor(target.values, dtype=torch.float32)

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]

def split_dataset(dataset, seed, train_size):
    torch.manual_seed(seed)
    # Dataset 전체 크기
    dataset_size = len(dataset)

    # 훈련, 검증, 테스트 데이터셋의 크기를 정의
    train_size = int(train_size * dataset_size)
    val_size = dataset_size - train_size 

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset