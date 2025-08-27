import streamlit as st
import spacy
import time

# ---------------------------
# Load spaCy Korean model
# ---------------------------
@st.cache_resource
def load_pipeline():
    # make sure model is installed: python -m spacy download ko_core_news_lg
    return spacy.load("ko_core_news_lg")

nlp = load_pipeline()

# ---------------------------
# Extract function
# ---------------------------
def extract_all_nouns_with_modifiers_spacy(sentence):
    start = time.time()
    doc = nlp(sentence)
    results = []

    # root token
    root = next((t for t in doc if t.head == t), None)

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:  # only nouns
            is_root = token == root or token.head == root
            modifiers, adpositions = [], []

            for child in token.children:
                # adjectival modifiers
                if child.dep_ in ["amod", "nummod", "compound"] or child.pos_ == "ADJ":
                    modifiers.append(child.text)
                # 조사 / adpositions
                if child.dep_ == "case" or child.pos_ == "ADP":
                    adpositions.append(child.text)

            results.append({
                "target_noun": token.text,
                "is_main_product": is_root,
                "adjectives": list(set(modifiers)),
                "adpositions": list(set(adpositions))
            })

    end = time.time()
    elapsed_ms = (end - start) * 1000
    return results, elapsed_ms

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🔍 Korean Noun Modifier Extractor (spaCy)")

sentence = st.text_input("Enter a Korean sentence:", "저렴한 휴대전화 보여 줘")

if st.button("Analyze"):
    with st.spinner("Parsing with spaCy..."):
        output, elapsed_time = extract_all_nouns_with_modifiers_spacy(sentence)

    st.subheader("Results")
    st.write(f"Execution Time: {elapsed_time:.2f} ms")
    st.json(output)
