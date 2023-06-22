import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

st.title('Apple batch evaluator - chat assistance')

if 'Batch_dict' not in st.session_state:
    st.write('Please upload a batch first, then return to this page')

else:
    model = SentenceTransformer('all-MiniLM-L12-v2')
    batch_dict = st.session_state['Batch_dict']

    user_question = st.text_input('Ask a question about the evaluated batch below')
    query_embedding = model.encode(user_question)
    
    possible_answers = [f'The total size of this batch is {sum(batch_dict.values())}',
                        f'The current date/time is {datetime.now()}',
                        f'I am not in a position to answer questions regarding topics other than apples']
    
    if len(batch_dict) == 2:
        possible_answers.extend([f'The number of bad apples in this batch is {batch_dict["rottenapples"]}',
                                 f'The number of good apples in this batch is {batch_dict["freshapples"]}',
                                 f'The percentages of good vs. bad apples in this batch are {(batch_dict["freshapples"] / sum(batch_dict.values()))*100} and {(batch_dict["rottenapples"] / sum(batch_dict.values()))*100} respectively.'])
    elif len(batch_dict) == 4:
        possible_answers.extend([f'This batch contains {batch_dict["Blotch_Apple"]} apples with blotch.',
                                 f'This batch contains {batch_dict["Rot_Apple"]} rotten apples.',
                                 f'This batch contains {batch_dict["Scab_Apple"]} apples with scab.',
                                 f'This batch contains {batch_dict["Normal_Apple"]} normal apples.'])
    
    answer_embedding = model.encode(possible_answers)
    similarity_scores = util.dot_score(query_embedding, answer_embedding)

    if user_question:
        st.write(possible_answers[similarity_scores.argmax()])
    

    

    