import streamlit as st
import spacy
from collections import Counter
import numpy as np

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Define POS tags to exclude
EXCLUDE_POS = {"PUNCT", "SYM", "DET", "CCONJ", "PART", "SCONJ"}

# Function to calculate POS entropy
def pos_entropy(pos_counts, total_tokens):
    probabilities = [count / total_tokens for count in pos_counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

# Function to analyze text
def analyze_text(text):
    doc = nlp(text)
    
    # Filter out unwanted POS before counting total tokens
    filtered_tokens = [token for token in doc if token.pos_ not in EXCLUDE_POS]
    total_tokens = len(filtered_tokens)
    
    # Count POS only from meaningful tokens
    pos_counts = Counter([token.pos_ for token in filtered_tokens])
    
    noun_count = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
    noun_percentage = (noun_count / total_tokens) * 100 if total_tokens > 0 else 0
    entropy = pos_entropy(pos_counts, total_tokens) if total_tokens > 0 else 0
    
    # Classify text
    if noun_percentage > 50 and entropy < 2.5:
        classification = "🔹 Keyword-heavy / Structured Query"
    else:
        classification = "🔹 General Text"
    
    return noun_percentage, entropy, classification, pos_counts

# Streamlit App
st.title("📊 Text Analysis: Noun Percentage & POS Diversity")
st.markdown("Analyze how noun-heavy and diverse your input text is.")

# User Input
user_input = st.text_area("Enter your text:", "Buy the best Samsung phone with 8GB RAM and 120Hz display.")

if st.button("Analyze"):
    if user_input.strip():
        noun_pct, entropy, classification, pos_counts = analyze_text(user_input)
        
        # Display Results
        st.subheader("📝 Analysis Result:")
        st.write(f"**Noun Percentage:** {noun_pct:.2f}%")
        st.write(f"**POS Diversity (Entropy):** {entropy:.2f}")
        st.write(f"**Classification:** {classification}")
        
        # Display POS Breakdown
        st.subheader("📌 POS Tag Breakdown:")
        st.write(dict(pos_counts))
    else:
        st.warning("⚠️ Please enter some text for analysis.")

# Footer
st.markdown("---")
st.markdown("🔹 **Noun Percentage**: Measures how noun-heavy the text is.")
st.markdown("🔹 **POS Diversity (Entropy)**: Measures how varied the sentence structure is.")
st.markdown("🔹 **Classification**: Determines if the input is a keyword-heavy query, conversational text, or balanced.")
