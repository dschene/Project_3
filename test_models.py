import torch
import torchvision.transforms as transforms
import json
import os
import matplotlib.pyplot as plt

from train_model import set_device
from torch.utils.data import DataLoader
from project_modules.DataClass import CustomDataset
from project_modules.DataClass import get_mean_and_std
from torchvision.datasets import ImageFolder
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay

path_to_testdata = '../data/No_background/Original/Test'

# transformation for getting the mean and std of the test set
test_transforms_formeanstd = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# making a loader for getting the mean and std of the test set
test_loader_formeanstd = DataLoader(ImageFolder(root=path_to_testdata, transform=test_transforms_formeanstd), 
                                    batch_size=16,
                                    shuffle=False)

# calling function for getting mean and std -> input for the 'actual' normalization transformation
mean, std = get_mean_and_std(test_loader_formeanstd)

# defining transformations on test data
test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# creating a custom class and dataloader for test data
test_dataset = CustomDataset(path_to_testdata, test_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#print(test_dataset.class_to_label)

n_of_classes = len(test_dataset.class_to_label)
label_list = list(test_dataset.class_to_label.keys())

# function for model evaluation
def eval_model(model, testloader):
    timestamp = datetime.now()

    model_metrics = {
    'Model type': None,
    'Timestamp': str(timestamp),
    'Model accuracy': None,
    'Model architecture': None
    }

    model.eval()
    device = set_device()
    t_correct = 0
    total = 0
    preds_list = torch.zeros(0,dtype=torch.long, device=device)
    labs_list = torch.zeros(0,dtype=torch.long, device=device)

    with torch.no_grad():
        
        for data in testloader:

            images, labels = data
            images = images.to(device)
            output = model(images)
            total += labels.size(0)
            _, predicted = torch.max(output.data, 1)
            t_correct += (predicted == labels).sum().item()

            preds_list=torch.cat([preds_list,predicted.view(-1)])
            labs_list=torch.cat([labs_list,labels.view(-1)])

    test_acc = (t_correct / len(test_dataset))*100

    model_metrics['Model accuracy'] = round(test_acc, 2)
    model_metrics['Test data'] = path_to_testdata
    model_metrics['Model architecture'] = str(model.state_dict)
    model_metrics['Number of classes'] = n_of_classes
    model_metrics['Model mapping'] = test_dataset.class_to_label

    print(f"Total images in test set: {total}")
    print(f"Total amount of correctly classified images: {t_correct}")
    print(f"Test Accuracy: {test_acc} /n")

    conf_mat = input("Show confusion matrix? Enter 'y' or 'n': ")

    if conf_mat == 'y':
        ConfusionMatrixDisplay.from_predictions(labs_list.numpy(), preds_list.numpy())
        plt.xticks(range(len(label_list)),label_list)
        plt.yticks(range(len(label_list)), label_list)
        plt.show()
    
    else:
        pass

    return model_metrics


while True:

    which_model = input('Choose the model for testing: enter \'r\' for resnet, \'c\' for custom: ')

    if which_model == 'r':
        model = torch.load('../models/resnet_model.pth')
        model_type = 'resnet'
        break

    elif which_model == 'c':
        model = torch.load('../models/custom_model.pth')
        model_type = 'custom'
        break 

    else:
        print('Invalid input')
        continue


def add_dict(dictionary, filename, model_type):
    dictionary['Model type'] = model_type
    if not os.path.isfile(filename):

        with open(filename, 'w') as f:
            json.dump([], f, indent=4, separators=(',', ': '))

    with open(filename, 'r') as f:
        data = json.load(f)

    data.append(dictionary)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))
        f.write('\n')


add_dict(eval_model(model, test_loader), '../models/models.json', model_type)
