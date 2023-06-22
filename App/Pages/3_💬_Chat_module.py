import streamlit as st

st.title('Apple batch evaluator - chat assistance')

if 'Batch_dict' not in st.session_state:
    st.write('Please upload a batch first, then return to this page')

else:
    batch_dict = st.session_state['Batch_dict']

    