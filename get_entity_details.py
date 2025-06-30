import streamlit as st
import spacy

# Load spaCy model once
nlp = spacy.load("en_core_web_md")

# Inference Function
def infer(sentence, target_noun):
    doc = nlp(sentence)
    root = [token for token in doc if token.head == token][0]

    related_adjs = []
    related_adps = []
    is_part_of_root_noun = False

    target_noun = target_noun.lower().strip()

    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower().strip()

        if target_noun in chunk_text:
            for token in chunk:
                if token == root or token.head == root:
                    is_part_of_root_noun = True

            for token in chunk:
                for child in token.children:
                    if child.pos_ == "ADJ":
                        related_adjs.append(child.text.lower().strip())
                    if child.pos_ == "ADP":
                        related_adps.append(child.text.lower().strip())

    related_adjs = list(set(related_adjs))
    related_adps = list(set(related_adps))

    return {
        "adj": related_adjs,
        "adp": related_adps,
        "is_main_product": is_part_of_root_noun
    }

# Streamlit App
st.set_page_config(page_title="NLP Noun Analyzer", page_icon="🔍", layout="centered")

st.title("🔍 GET ENTITY DETAILS")

st.write("Enter a query and target noun to analyze relationships using spaCy:")

with st.form(key='infer_form'):
    query = st.text_input("Enter your query", value="show me cheap samsung s24 with compatible charger")
    target_noun = st.text_input("Enter the target noun", value="samsung s24")
    submit_button = st.form_submit_button(label='Analyze')

if submit_button:
    with st.spinner("Analyzing..."):
        result = infer(query, target_noun)
    st.success("Analysis Complete ✅")
    st.subheader("Results:")
    st.json(result)

st.markdown("---")
st.caption("Powered by spaCy and Streamlit 🚀")
