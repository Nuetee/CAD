import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_features, 64)
        self.bn1 = nn.BatchNorm1d(64)  # 첫 번째 선형 레이어 후에 배치 정규화 적용
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, input_features)
        self.bn2 = nn.BatchNorm1d(input_features)  # 두 번째 선형 레이어 후에 배치 정규화 적용

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)  # 첫 번째 배치 정규화 적용
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)  # 두 번째 배치 정규화 적용
        out += identity
        out = self.relu(out)
        return out

class ResidualNet(nn.Module):
    def __init__(self, n_layers, in_dim, hidden_dim, out_dim):
        super(ResidualNet, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        layers = []
        layers.append(nn.Linear(self.in_dim, self.hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(self.n_layers - 1):
            layers.append(ResidualBlock(self.hidden_dim))

        layers.append(nn.Linear(self.hidden_dim, self.out_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out
