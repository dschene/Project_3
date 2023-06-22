import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Multipage App",
    page_icon="👋",
)

st.title('Welcome')
st.subheader('Start by uploading a batch of pictures by clicking on the \"Batch evaluator\" in the side bar')
im = Image.open('../project_modules/orchards.jpg')
st.image(im)

st.sidebar.success("Select a page above.")