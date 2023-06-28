import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch

from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from train_model import train_model_func
from train_model import set_device
from project_modules.DataClass import get_mean_and_std


# path to train data
train_data_path = '../data/Original_cropped_aug_extended/Train'

# transformation for getting the mean and std of the train set
test_transforms_formeanstd = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# making a loader for getting the mean and std of the test set
test_loader_formeanstd = DataLoader(ImageFolder(root=train_data_path, transform=test_transforms_formeanstd), 
                                    batch_size=16,
                                    shuffle=False)

# calling function for getting mean and std -> input for the 'actual' normalization transformation
mean, std = get_mean_and_std(test_loader_formeanstd)

# transformations to be applied to train data
train_transforms = transforms.Compose([
    #transforms.RandAugment(2,14),
    transforms.Resize((224, 224)),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomVerticalFlip(p=0.5),
    #transforms.RandomRotation(degrees=(30, 70)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# creating a DataLoader using ImageFolder
train_data = ImageFolder(train_data_path, transform=train_transforms)
train_loader = DataLoader(train_data, batch_size=50, shuffle=True)


resnet18_model = resnet18(pretrained=False)

num_features_18 = resnet18_model.fc.in_features
num_of_classes = len(train_data.class_to_idx)

resnet18_model.fc = nn.Linear(num_features_18, num_of_classes)


device = set_device()

resnet_18_model = resnet18_model.to(device)

model = resnet_18_model

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# run training
train_model_func(model, train_loader, loss_function, optimizer, 10)

# save model
save_model_path = f'../models/resnet_model.pth'
torch.save(model, save_model_path)