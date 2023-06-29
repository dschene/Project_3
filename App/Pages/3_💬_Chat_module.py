import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

st.title('Apple batch evaluator - chat support')

if 'Batch_dict' not in st.session_state:
    st.write('Please upload a batch first, then return to this page')

else:
    model = SentenceTransformer('all-MiniLM-L12-v2')
    batch_dict = st.session_state['Batch_dict']

    user_question = st.text_input('Ask a question about the evaluated batch below')
    query_embedding = model.encode(user_question)
    
    possible_answers = [f'The total size of this batch is {sum(batch_dict.values())}.',
                        f'The current date/time is {datetime.now()}.',
                        f'I am not in a position to answer questions about topics other than apples.']
    
    if len(batch_dict) == 2:
        possible_answers.extend([f'The number of bad apples in this batch is {batch_dict["rottenapples"]}',
                                 f'The number of good apples in this batch is {batch_dict["freshapples"]}',
                                 f'The percentages of good and bad apples in this batch are {(batch_dict["freshapples"] / sum(batch_dict.values()))*100} and {(batch_dict["rottenapples"] / sum(batch_dict.values()))*100} respectively.'])
    elif len(batch_dict) == 4:
        possible_answers.extend([f'This batch contains {batch_dict["Blotch_Apple"]} apples with blotch.',
                                 f'This batch contains {batch_dict["Rot_Apple"]} rotten apples.',
                                 f'This batch contains {batch_dict["Scab_Apple"]} apples with scab.',
                                 f'This batch contains {batch_dict["Normal_Apple"]} normal apples.',
                                 "Blotch fungus on apples is a common disease caused by a variety of fungi throughout the fruiting season. Fortunately, it’s a problem that’s limited to the skin of the apple. It’s also safe to eat unless you have a mold allergy, so for many homeowners, apple blotch fungus disease may not pose a serious enough threat to treat. For others, some level of treatment between none and orchard-level protection may seem more appropriate. Apple blotch symptoms usually present as quarter inch (6 mm.) or larger, irregular areas on the surface of infected fruits. The color may be cloudy or sooty, often making the apple surface appear olive green. It’s common for smaller areas to come together to form larger, non-circular spots on the skin. Apple blotch fungus disease is sometimes accompanied by a similar fungal disease known as “flyspeck,” which will add small, raised black spots in addition to the sooty blotches.",
                                 "Apple scab is a common fungal disease affecting the leaves and fruit to the point where the tree loses its leaves, and the apples are so blemished that they become unfit for eating. The apple scab fungus thrives in areas with lots of rain and high humidity, and during a warm, wet spring. Because there is no treatment for infected trees, early identification and prevention are crucial for its control."
        ])
    
    answer_embedding = model.encode(possible_answers)
    similarity_scores = util.dot_score(query_embedding, answer_embedding)

    if user_question:
        st.write(possible_answers[similarity_scores.argmax()])