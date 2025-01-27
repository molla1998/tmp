import streamlit as st
import spacy
import re
import random

# Load spaCy model for noun and dependency parsing
nlp_spacy = spacy.load("en_core_web_sm")

# List of common pronouns to filter out
PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", 
            "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs"}

def extract_keywords_in_order(query):
    # Preprocess the query
    query_lower = query.lower()

    # Extract noun phrases using spaCy (focus on nouns)
    doc = nlp_spacy(query)
    
    # Extract noun chunks (e.g., "phone", "tv", "laptop") and individual nouns
    noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]

    # Also consider individual nouns (since "tv" and "phone" may not be in noun chunks)
    nouns = [token.text.lower() for token in doc if token.pos_ == "NOUN"]

    # Combine noun phrases and individual nouns and remove duplicates
    combined_nouns = list(set(noun_phrases + nouns))

    # Extract numeric and alphanumeric patterns using regex (e.g., "50k", "8GB", "50MP")
    numeric_patterns = [
        match.group() for match in re.finditer(r"(\d+k|\d+\s?gb|\d+\s?mp|\d+\s?hz|₹?\d{1,3},?\d{1,3})", query_lower)
    ]

    # Combine all extractions and filter out pronouns
    filtered_keywords = list(set(combined_nouns + numeric_patterns))  # Remove duplicates

    # Return only relevant keywords
    return [keyword for keyword in filtered_keywords if keyword not in PRONOUNS]

# Streamlit App
st.title("Keyword Extractor")
st.markdown("Enter a query to extract keywords (nouns, numeric values, and features). Keywords will be highlighted in different transparent colors.")

# Input box for the query
query = st.text_input("Enter your query:", "")

# Button to extract keywords
if st.button("Extract"):
    if query.strip():
        # Extract keywords
        keywords = extract_keywords_in_order(query)
        
        # Display keywords in different transparent colors
        st.markdown("### Extracted Keywords:")
        for keyword in keywords:
            color = f"rgba({random.randint(50, 255)}, {random.randint(50, 255)}, {random.randint(50, 255)}, 0.5)"
            st.markdown(f'<span style="background-color: {color}; padding: 5px; border-radius: 5px; margin: 5px;">{keyword}</span>', unsafe_allow_html=True)
    else:
        st.warning("Please enter a query to extract keywords.")
