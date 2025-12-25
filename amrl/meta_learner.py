import torch
import torch.nn as nn

class MetaLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(6, 64, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.alpha = nn.Linear(16, 1)
        self.gamma = nn.Linear(16, 1)
        self.beta  = nn.Linear(16, 1)

    def forward(self, x, h=None):
        o, h = self.lstm(x, h)
        z = self.fc(o[:, -1])
        return torch.cat([
            torch.tanh(self.alpha(z)) * 0.30,
            torch.tanh(self.gamma(z)) * 0.02,
            torch.tanh(self.beta(z))  * 0.05
        ], dim=1), h
