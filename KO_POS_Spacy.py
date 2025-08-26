import spacy
import re

# ---------------------------
# Load spaCy Korean model
# ---------------------------
# install first if missing:
# !python -m spacy download ko_core_news_lg
nlp = spacy.load("ko_core_news_lg")

# ---------------------------
# Extract function
# ---------------------------
def extract_all_nouns_with_modifiers_spacy(sentence):
    doc = nlp(sentence)
    results = []

    # root token
    root = next((t for t in doc if t.head == t), None)

    for token in doc:
        # only consider nouns / proper nouns
        if token.pos_ in ["NOUN", "PROPN"]:
            is_root = token == root or token.head == root

            modifiers, adpositions = [], []

            # children of this token
            for child in token.children:
                # adjectival modifiers
                if child.dep_ in ["amod", "nummod", "compound"] or child.pos_ == "ADJ":
                    modifiers.append(child.text)
                # 조사 / adpositions
                if child.dep_ == "case" or child.pos_ == "ADP":
                    adpositions.append(child.text)

            results.append({
                "target_Noun": token.text,
                #"target_lemma": token.lemma_,
                "is_main_product": is_root,
                "adjectives": list(set(modifiers)),
                "adpositions": list(set(adpositions))
            })

    return results


# ---------------------------
# Example test
# ---------------------------
sentence = "저렴한 휴대전화 보여 줘"
print(extract_all_nouns_with_modifiers_spacy(sentence))
