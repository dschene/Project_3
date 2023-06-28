import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(24, 44, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(44 * 56 * 56, 64)
        self.fc2 = nn.Linear(64, 96)
        self.fc3 = nn.Linear(96, self.num_classes)

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


class CNNModel2(nn.Module):
    def __init__(self, num_of_classes):
        super(CNNModel2, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(12,24, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(24, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(256 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,num_of_classes)
        )
        
    def forward(self, xb):

        return self.network(xb)