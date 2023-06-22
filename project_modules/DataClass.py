import os
import matplotlib.pyplot as plt
import random
import torchvision

from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transforms = transform
        self.class_to_label = {}
        self.images = []
        self.labels = []

        class_names = [d for d in sorted(os.listdir(folder_path)) if os.path.isdir(os.path.join(folder_path, d))]
        
        for i, class_name in enumerate(class_names):
            class_path = os.path.join(folder_path, class_name)

            self.class_to_label[class_name] = i
                
            for image_name in os.listdir(class_path):
                if image_name.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_path, image_name)
                    self.images.append(image_path)
                    self.labels.append(i)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        
        image = self.images[index]
        label = self.labels[index]
        
        image = Image.open(image).convert("RGB")
        
        if self.transforms is not None:
            image = self.transforms(image)
            
        return image, label
    
    def display_image_from_tensor(self):
        random_index = random.randint(0, len(self.images) - 1)
        
        img_path = self.images[random_index]
        img_label = self.labels[random_index]
        img = torchvision.io.read_image(img_path).permute(1, 2, 0)
        
        for key, value in self.class_to_label.items():
            if value == img_label:
                class_name = key
                
        plt.imshow(img)
        plt.title(f"Class: {class_name}")
        plt.axis('off')
        plt.show() 


# function for getting the mean and std of a dataset
def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_a_batch
    
    mean /= total_images_count
    std /=total_images_count
    
    return mean, std