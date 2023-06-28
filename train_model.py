import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from project_modules.DataClass import CustomDataset, get_mean_and_std
from project_modules.CNN import CNNModel, CNNModel2
from tqdm import tqdm

# training data directory
path_to_traindata = '../data/Original_cropped_wo_28/Train'

# function for displaying an img
def display_image_from_tensor(tensor):
    tensor = np.transpose(tensor, (1, 2, 0))
    plt.imshow(tensor)
    plt.axis('off')
    plt.show()

#function for setting the device
def set_device():
    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
    
    return torch.device(dev)

# transformation for getting the mean and std of the train set
train_transforms_formeanstd = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# making a loader for getting the mean and std of the test set
test_loader_formeanstd = DataLoader(ImageFolder(root=path_to_traindata, transform=train_transforms_formeanstd), 
                                    batch_size=16,
                                    shuffle=False)

# calling function for getting mean and std -> input for the 'actual' normalization transformation
mean, std = get_mean_and_std(test_loader_formeanstd)
mean_2, std_2 = [0.6327, 0.5601, 0.4343], [0.2058, 0.2333, 0.2498]
# transformations to be applied to training data

N, M = 2, 14

train_transforms = transforms.Compose([
    #transforms.RandomCrop(32, 4),
    transforms.Resize((224, 224)),
    #transforms.RandAugment(N,M),
    #transforms.RandomVerticalFlip(p=0.5),
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomRotation(degrees=(30, 70)),
    transforms.ToTensor(),
    transforms.Normalize(mean_2, std_2),
])

# option 1: creating a custom dataset with the Dataset() class
train_dataset_from_c = CustomDataset(path_to_traindata, train_transforms)
# # making a DataLoader object for the training data
train_dataloader_from_c = DataLoader(train_dataset_from_c, batch_size=32, shuffle=True)

# option 2: creating a dataset using ImageFolder() 
train_dataset_from_f = ImageFolder(path_to_traindata, transform=train_transforms)
# # making a DataLoader object for the training data
train_dataloder_from_f = DataLoader(train_dataset_from_f, batch_size=32, shuffle=True)

# both variations have the same class-to-label mapping:
(train_dataset_from_c.class_to_label == train_dataset_from_f.class_to_idx) # == True

# creating a model object with 2 or 4 possible output classes
n_of_classes = len(train_dataset_from_c.class_to_label)
apples_model = CNNModel(n_of_classes).to(set_device())
apples_model_2 = CNNModel2(n_of_classes).to(set_device())
# defining loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(apples_model.parameters(), lr=0.0001)


# function for training the model
def train_model_func(model, trainloader, criterion, optimizer, n_epochs):

    device = set_device()
    model = model.to(device)
    model.train()
    
    for n in range(n_epochs):
        print(f'Current epoch: {n+1}')
        
        running_loss = 0
        running_correct = 0
        total_predicted = 0
        
        for data in tqdm(trainloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            total_predicted += labels.size(0)
            
            optimizer.zero_grad()
            output = model(images)
            
            _, predicted = torch.max(output.data, 1)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()
        
        epoch_loss = round(running_loss / len(trainloader), 2)
        epoch_acc = round(100.0 * running_correct / total_predicted, 2)
        
        
        print(f'TRAINING SET - {running_correct} out of {total_predicted} correct. Epoch accuracy is {epoch_acc}. Epoch loss is {epoch_loss}')

    print(f'Final metrics of trained model: accuracy is {epoch_acc}, loss is {epoch_loss}')


# run training
if __name__ == "__main__":
    train_model_func(apples_model, train_dataloader_from_c, loss_function, optimizer, 15)

    # store model in 'models' directory
    save_model_path = '../models/custom_model.pth'

    #model als geheel opslaan, niet enkel state_dict
    torch.save(apples_model, save_model_path)