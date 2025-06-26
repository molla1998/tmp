import streamlit as st
import spacy
import time

nlp = spacy.load("en_core_web_md")

# Define connectors with grammar-based rules
left_primary = {"with", "for", "under", "by", "on"}
right_primary = set()

noun_pos_tags = {"NOUN", "PROPN", "NUM"}

def extract_info(text):
    doc = nlp(text)

    primary = {"main_noun": "", "prev_nouns": "", "adj": "", "adp": ""}
    secondary = {"main_noun": "", "next_nouns": "", "adj": "", "adp": ""}

    connector_token = None

    for token in doc:
        if token.text.lower() in left_primary.union(right_primary) and token.pos_ == "ADP":
            connector_token = token
            break

    if connector_token:
        left_tokens = list(doc[:connector_token.i])
        right_tokens = list(doc[connector_token.i + 1:])

        left_main_noun, left_prev_nouns = "", ""
        right_main_noun, right_next_nouns = "", ""

        for i in reversed(range(len(left_tokens))):
            token = left_tokens[i]
            if token.pos_ in noun_pos_tags:
                left_main_noun = token.text
                prev_nouns = []
                for j in reversed(range(i)):
                    if left_tokens[j].pos_ in noun_pos_tags:
                        prev_nouns.insert(0, left_tokens[j].text)
                    else:
                        break
                left_prev_nouns = " ".join(prev_nouns)
                break

        for i, token in enumerate(right_tokens):
            if token.pos_ in noun_pos_tags:
                right_main_noun = token.text
                next_nouns = []
                for j in range(i + 1, len(right_tokens)):
                    if right_tokens[j].pos_ in noun_pos_tags:
                        next_nouns.append(right_tokens[j].text)
                    else:
                        break
                right_next_nouns = " ".join(next_nouns)
                break

        connector = connector_token.text.lower()

        if connector in left_primary:
            primary["main_noun"] = left_main_noun
            primary["prev_nouns"] = left_prev_nouns
            secondary["main_noun"] = right_main_noun
            secondary["next_nouns"] = right_next_nouns
        elif connector in right_primary:
            primary["main_noun"] = right_main_noun
            primary["prev_nouns"] = right_next_nouns
            secondary["main_noun"] = left_main_noun
            secondary["next_nouns"] = left_prev_nouns

        for token in doc:
            if token.head.text == primary["main_noun"]:
                if token.pos_ == "ADJ":
                    primary["adj"] = token.text
                if token.pos_ == "ADP":
                    primary["adp"] = token.text

            if token.head.text == secondary["main_noun"]:
                if token.pos_ == "ADJ":
                    secondary["adj"] = token.text
                if token.pos_ == "ADP":
                    secondary["adp"] = token.text

    return {
        "Primary noun": primary["prev_nouns"] + " " + primary["main_noun"],
        "Adj ref Primary noun": primary["adj"],
        "Secondary noun": secondary["main_noun"]+ " " +secondary["next_nouns"],
        "Adj ref Secondary noun": secondary["adj"]
    }

# Streamlit UI
st.title("Noun & Attribute Extractor")
st.write("This app extracts primary & secondary nouns and related attributes based on grammar rules.")

text = st.text_input("Enter your sentence:", "deals on galaxy s24")

if st.button("Extract Info"):
    start_time = time.time()
    result = extract_info(text)
    end_time = time.time()

    st.success("Extraction Completed!")

    st.write("### Extraction Results:")
    for key, value in result.items():
        st.write(f"**{key}:** {value if value else '-'}")

    st.write(f"\n**Inference Time:** {round((end_time - start_time) * 1000, 2)} ms")
