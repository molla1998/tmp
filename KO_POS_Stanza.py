import streamlit as st
import stanza

# ---------------------------
# Load Stanza Korean pipeline
# ---------------------------
@st.cache_resource
def load_pipeline():
    stanza.download("ko")  # Run only once locally
    return stanza.Pipeline("ko")

nlp = load_pipeline()
# Print pipeline processors
print("Processors in pipeline:", nlp.processors.keys())

# Check each processor's model type
for processor, obj in nlp.processors.items():
    print(f"{processor}: {type(obj)}")
    
# ---------------------------
# Extract function
# ---------------------------
def extract_all_nouns_with_modifiers(sentence):
    start = time.time()
    doc = nlp(sentence)
    results = []

    for sent in doc.sentences:
        root = next((w for w in sent.words if w.head == 0), None)

        for word in sent.words:
            # only nouns / proper nouns
            if word.upos in ["NOUN", "PROPN"]:
                is_root = word.deprel == "root" or (
                    word.head > 0 and sent.words[word.head - 1] == root
                )

                modifiers, adpositions = [], []

                for child in sent.words:
                    if child.head == word.id:
                        if child.deprel in ["amod", "nummod", "compound"] or child.upos == "ADJ":
                            modifiers.append(child.text)
                        if child.deprel == "case" or child.upos == "ADP":
                            adpositions.append(child.text)

                results.append({
                    "target_noun": word.text,
                    #"target_lemma": word.lemma,
                    "is_main_product": is_root,
                    "adjectives": list(set(modifiers)),
                    "adpositions": list(set(adpositions))
                })
        end = time.time() 
    
    elapsed_time_ms = (end - start) * 1000  # convert to ms
    
    return results,elapsed_time_ms

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🔍 Korean Noun Modifier Extractor")

sentence = st.text_input("Enter a Korean sentence:", "저렴한 휴대전화 보여 줘")

if st.button("Analyze"):
    with st.spinner("Parsing with Stanza..."):
        output,elapsed_time_ms = extract_all_nouns_with_modifiers(sentence)
    print(f"Execution Time: {exec_time_ms:.3f} ms")
    st.subheader("Results")
    st.write(f"Execution Time: {exec_time_ms:.3f} ms")
    st.json(output)
