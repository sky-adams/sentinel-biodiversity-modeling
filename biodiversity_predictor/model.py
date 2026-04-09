import torch
import torch.nn as nn


class BIIRegressor(nn.Module):
    def __init__(self, in_channels=4, base_channels=32, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 2, base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)
