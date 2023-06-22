import torch.nn as nn
import sys
import os

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 14, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(14, 28, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(28 * 56 * 56, 50)
        self.fc2 = nn.Linear(50, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
    
        return x

