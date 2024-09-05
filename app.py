import streamlit as st
import openai
import math
import dill
import os

POS_TAGGER_MODEL_FILE = "pos-tagger.pkl"


def hmm_pos_tagger(sentence):
    words = [word for word in sentence.split()]
    last_ch = words[-1][-1]
    if not last_ch.isalnum() and len(words[-1]) > 1:
        words[-1] = words[-1][:-1]
        words.append(last_ch)

    with open("pos-tagger.pkl", "rb") as f:
        model = dill.load(f)
        print(sentence)
        return model.predict(' '.join(words))

openai.api_key = os.environ.get("OPENAI_API_KEY")
def gpt4_pos_tagger(sentence):
    allowed_tags = "{'X', 'ADV', 'PRT', 'CONJ', 'ADP', 'VERB', 'PRON', 'ADJ', 'NOUN', '.', 'NUM', 'DET'}"
    prompt = (
        f"Tag the parts of speech for the following sentence using only these tags {allowed_tags}. "
        f"These tags are same as the tags from the universal tag set of the brown corpus."
        f"If a word does not fit any category, use 'X'.\n\n"
        f"Sentence: '{sentence}'\n\n"
        "Output only the tags in the same order as the words, separated by spaces."
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=5000
    )
    
    tags = response['choices'][0]['message']['content'].strip().split()

    return tags

def create_comparison_table(sentence_words, gpt4_tags, hmm_tags):

    table = "<table style='width:100%; border: 1px solid black; border-collapse: collapse;'>"

    table += "<tr><th style='border: 1px solid black; padding: 8px; text-align: left;'>Sentence</th>"
    for word in sentence_words:
        table += f"<td style='border: 1px solid black; padding: 8px;'>{word}</td>"
    table += "</tr>"


    table += "<tr><th style='border: 1px solid black; padding: 8px; text-align: left;'>GPT-4</th>"
    for tag in gpt4_tags:
        table += f"<td style='border: 1px solid black; padding: 8px; color: orange;'>{tag}</td>"
    table += "</tr>"


    table += "<tr><th style='border: 1px solid black; padding: 8px; text-align: left;'>HMM POS Tagger</th>"
    ind = 0
    for tag in hmm_tags:
        color = "green" if tag.strip() == gpt4_tags[ind].strip() else "red"
        table += f"<td style='border: 1px solid black; padding: 8px;'><span style=\"color:{color}\">{tag}</span></td>"
        ind += 1
    table += "</tr>"

    table += "</table>"
    
    return table


st.set_page_config(page_title="POS Tagger Demo", page_icon="📊", layout="wide")
# st.title("POS Tagger Comparison: HMM vs GPT-4")
st.markdown("<h1 style='text-align: center;'>POS Tagger Comparison: HMM vs GPT-4</h1>", unsafe_allow_html=True)




sentence = st.text_area("", height=50,  placeholder="Type your sentence here...")


   
if st.button("Submit",use_container_width=True):
    if sentence.strip():
        sentence_words = sentence.split()
        gpt4_tags = gpt4_pos_tagger(sentence)
        hmm_tags = hmm_pos_tagger(sentence)
        for i in range(len(gpt4_tags)):
            actual_tags = ''.join([c for c in gpt4_tags[i] if c != "'"])
            gpt4_tags[i] = actual_tags
    
        comparison_table = create_comparison_table(sentence_words, gpt4_tags, hmm_tags)
    
        st.write("### POS Tagging Comparison")
        st.markdown(comparison_table, unsafe_allow_html=True)
    else:
        st.warning("Please enter a sentence.")
