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

st.write("Enter a query and one or more target nouns (comma-separated) to analyze relationships using spaCy:")

with st.form(key='infer_form'):
    query = st.text_input("Enter your query", value="show me cheap samsung s24 with compatible charger and iphone 15 case")
    target_nouns_input = st.text_input("Enter target nouns (comma-separated)", value="samsung s24, iphone 15 case")
    submit_button = st.form_submit_button(label='Analyze')

if submit_button:
    target_nouns = [noun.strip() for noun in target_nouns_input.split(",") if noun.strip()]
    
    if not target_nouns:
        st.warning("Please enter at least one valid target noun.")
    else:
        with st.spinner("Analyzing..."):
            for idx, noun in enumerate(target_nouns, 1):
                result = infer(query, noun)
                with st.expander(f"Result {idx}: '{noun}'"):
                    st.json(result)
                    
        st.success("Analysis Complete ✅")

st.markdown("---")
st.caption("Powered by spaCy and Streamlit 🚀")
