import os
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image

def apply_transforms(main_folder_path, transformations):

    for folder_name in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder_name)

        if os.path.isdir(folder_path):

            for filename in os.listdir(folder_path):

                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(folder_path, filename)
                    print(file_path)
                    img = Image.open(file_path)

                    for i in range(5):
                        image_tensor = transformations(img)
                        path_to_save = folder_path + "/augmented_" + str(i) + '_' + filename
                        save_image(image_tensor.float(), path_to_save)


transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.RandomPerspective(distortion_scale=0.6, p=0.75),
    transforms.Resize((224,224))
])

if __name__ == "__main__":

    apply_transforms('../data/Cropped_v2/Train', transformations)

