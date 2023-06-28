import os
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image

def apply_transforms(main_folder_path, standard_transforms, var_transforms):

    for folder_name in os.listdir(main_folder_path):
        folder_path = os.path.join(main_folder_path, folder_name)
    
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):

                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(folder_path, filename)
                    img = Image.open(file_path)
                    img_t = standard_transforms(img)

                    for i, t in enumerate(var_transforms):
                        var_img_t = t(img_t)
                        save_path = f'{folder_path}/' + f"_{str(i)}_" + filename
                        save_image(var_img_t.float(), save_path)
                    
                    


transformations_all = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))
])

transformations_var = [
    transforms.RandomHorizontalFlip(p=1), 
    transforms.RandomVerticalFlip(p=1),
    transforms.RandomRotation(degrees=(0, 180))
    #transforms.RandomPerspective(distortion_scale=0.6, p=1)
    ]

if __name__ == "__main__":

    apply_transforms('../data/Original_cropped_wo_28/Test', transformations_all, transformations_var)

