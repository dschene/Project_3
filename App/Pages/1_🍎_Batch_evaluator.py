import streamlit as st
import torch
import json
import streamlit as st
from PIL import Image
from torchvision import transforms


st.title('Apple batch evaluator')

best_model_metrics = max(json.load(open('../models/models.json')), key=lambda x: x['Timestamp'])

model = torch.load('../models/resnet_model.pth')

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



st.subheader(f"Latest model dates from {best_model_metrics['Timestamp']} and has an accuracy of {best_model_metrics['Model accuracy']}%")

uploaded_files = st.file_uploader("Upload apple batch images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:

    pil_images = [Image.open(file) for file in uploaded_files]

    with st.spinner('Evaluating batch'):
        batch = process_image(pil_images)
        st.session_state['Batch_dict'] = batch

    st.success('Batch evaluated - please head over the the AQL section for batch metrics.')
