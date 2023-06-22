import streamlit as st
import sys
import os
sys.path.append(os.path.abspath("../.."))
from Final.project_modules.Aql import Aql

Aql_object = Aql()

def a_or_j(max_a, bad_a):

    if bad_a <= max_a:
        return 'accepted'
    else:
        return 'rejected'
    
st.title('Acceptance Quality Levels')

if 'Batch_dict' not in st.session_state:
    st.write('Please upload a batch first, then return to this page.')

else:
    batch = st.session_state['Batch_dict']
    total_size = sum(batch.values())
    aql_dict = Aql_object.get_aql_for_batchsize(total_size)

    st.subheader("Default inspection level is set to I")
    
    st.write(f'Uploaded batch of size {total_size} contains:')
    
    for k, v in st.session_state['Batch_dict'].items():
        st.write(f'{v} instances of {k}')
    
    if st.button("Perform AQL inspection"):
            if len(batch) == 2:
                bad_apples = batch['rottenapples']

            else:
                bad_apples = batch['Scab_Apple'] + batch['Blotch_Apple'] + batch['Rot_Apple']

            st.write(f'A sample size of {total_size} has the following acceptance quality levels: ')

            for k, v in aql_dict.items():
                st.write(f'AQ level {k} - According to this inspection level the current batch is {a_or_j(v, bad_apples).upper()}')

            # with st.spinner('Performing AQL inspection'):

            #     if bad_apples == 0:
            #         st.write('Apple batch is of class 1 and is suited for sale in supermarkets or grocery stores.')
                
            #     elif bad_apples <= 3:
            #         st.write('Apple batch is of class 2 and is suited for apple sauce production.')

            #     elif 3 < bad_apples <= 7:
            #         st.write('Apple batch is of class 3 and is suited for syrup production.')

            #     else:
            #         st.write('Apple batch contains too many bad apples and is therefore rejected.')




