# Imports
import torch
import json
import streamlit as st
from PIL import Image
from torchvision import transforms

#latest moet nog worden gewijzigd in 'best'
best_model_metrics = max(json.load(open('../models/models.json')), key=lambda x: x['Model accuracy'])

if best_model_metrics['Model type'] == 'resnet':

    model = torch.load('../models/resnet_model.pth')
    
else:
    model = torch.load('../models/custom_model.pth')

def process_image(image_batch):
    model.eval()
    mapping = best_model_metrics['Model mapping']

    reversed_mapping = {v: k for k,v in mapping.items()}

    convert_tensor = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        ])

    tensor_list = [convert_tensor(p) for p in image_batch]
    tensor_batch = torch.stack(tensor_list)
    output = model(tensor_batch)

    _, predicted = torch.max(output.data, 1)

    counter_dict = {k: 0 for k in mapping.keys()}
    predicted_list = [reversed_mapping[v] for v in predicted.tolist()]

    for a in predicted_list:
        counter_dict[a] += 1

    return counter_dict


def main():
    st.title("Apple batch evaluator")
    st.subheader(f"Best performing {best_model_metrics['Model type']} model dates from {best_model_metrics['Timestamp']} and has an accuracy of {best_model_metrics['Model accuracy']}%")

    uploaded_files = st.file_uploader("Upload apple batch images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:

        pil_images = [Image.open(file) for file in uploaded_files]

        with st.spinner('Evaluating batch'):
            batch = process_image(pil_images)

        st.success('Batch evaluated')
        st.write('Batch contains:')
        
        for k, v in batch.items():
            st.write(f'{v} instances of {k}')

        if st.button("Perform AQL inspection"):
            st.write('General Inspection Level is set to I by default')
            
            if len(batch) == 2:

                bad_apples = batch['rottenapples']

            else:

                bad_apples = batch['Scab_Apple'] + batch['Blotch_Apple'] + batch['Rot_Apple']

            with st.spinner('Performing AQL inspection'):

                if bad_apples == 0:
                    st.write('Apple batch is of class 1 and is suited for sale in supermarkets or grocery stores.')
                
                elif bad_apples <= 3:
                    st.write('Apple batch is of class 2 and is suited for apple sauce production.')

                elif 3 < bad_apples <= 7:
                    st.write('Apple batch is of class 3 and is suited for syrup production.')

                else:
                    st.write('Apple batch contains too many bad apples and is therefore rejected.')
            

main()