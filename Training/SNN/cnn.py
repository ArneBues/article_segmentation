import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_mask = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )

        n_channels_mask = self.cnn_mask(torch.empty(1, 3, 256, 256)).size(-1)
        print(self.cnn_mask(torch.empty(1, 3, 256, 256)).size())
        self.fc_mask = nn.Sequential(
            nn.Flatten(),
        )

    def forward(self, input):
        output = self.cnn_mask(input)
        return output
