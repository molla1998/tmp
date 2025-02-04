import spacy
import re
import streamlit as st
from collections import Counter

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Expanded regex pattern for electronics specifications
spec_patterns = r"\b\d+(\.\d+)?\s?(gb|mb|tb|mp|hz|mah|inch|cm|w|ghz|nm|fps|ppi|v|a|cores|k|dpi|bits|ms|ohm|db|nits|lux|g|kg|mbps|ghz|nm|m|s)\b"

# Function to extract keywords and noun chunks
def extract_keywords(query):
    query_lower = query.lower()  # Convert query to lowercase
    doc = nlp(query_lower)  # Process query with spaCy

    # Extract NER-based entities (brands, product types, money, quantities)
    ner_keywords = [ent.text for ent in doc.ents if ent.label_ in {"ORG", "PRODUCT", "MONEY", "QUANTITY"}]

    # Extract POS-based keywords (NOUN & PROPN)
    pos_keywords = [token.text for token in doc if token.pos_ in {"NOUN", "PROPN"} and len(token.text) > 2]

    # Extract electronic specs using regex
    spec_keywords = re.findall(spec_patterns, query_lower, re.IGNORECASE)
    spec_keywords = [''.join(match) for match in spec_keywords]  # Join regex tuples

    # Extract noun chunks (multi-word phrases)
    chunk_keywords = [chunk.text for chunk in doc.noun_chunks]

    # Combine individual keywords & remove duplicates
    individual_keywords = list(set(ner_keywords + pos_keywords + spec_keywords))
    noun_chunk_keywords = list(set(chunk_keywords))

    return individual_keywords, noun_chunk_keywords

# Streamlit App
st.title("🔍 Searchable Keyword Extractor")
st.markdown("Extracts important **keywords** and **multi-word phrases** from a query.")

# Input Box
user_input = st.text_area("Enter your query:", "Show me a Samsung Galaxy S phone with 8GB RAM, 120Hz display, and 50MP camera.")

# Submit Button
if st.button("Extract Keywords"):
    if user_input.strip():
        keywords, noun_chunks = extract_keywords(user_input)
        
        # Display Results
        st.subheader("📝 Extracted Keywords:")
        st.write(", ".join(keywords) if keywords else "No individual keywords found.")

        st.subheader("🔗 Extracted Noun Chunks (Multi-word phrases):")
        st.write(", ".join(noun_chunks) if noun_chunks else "No noun chunks found.")
    else:
        st.warning("⚠️ Please enter a query.")

# Footer
st.markdown("---")
st.markdown("🔹 **NER-based Extraction**: Identifies brands, product types, quantities, and prices.")
st.markdown("🔹 **POS-based Extraction**: Extracts meaningful nouns & proper nouns for searchability.")
st.markdown("🔹 **Regex-based Extraction**: Finds electronics-related specs like RAM, refresh rate, resolution, etc.")
st.markdown("🔹 **Noun Chunk Extraction**: Captures meaningful multi-word phrases (e.g., 'Samsung Galaxy S series phone').")
