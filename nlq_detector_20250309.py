import streamlit as st
import spacy
import numpy as np
import re
from collections import Counter

# Load spaCy model
nlp = spacy.load("en_core_web_md")

# Define POS tags to exclude
EXCLUDE_POS = {"PUNCT", "SYM"}

# Function to calculate Shannon entropy
def pos_entropy(pos_counts, total_tokens):
    probabilities = [count / total_tokens for count in pos_counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

# Function to analyze text
def analyze_text(text):
    doc = nlp(text)
    
    # List to store token-POS pairs
    tagged_tokens = []
    
    # Custom POS tagging: Treat numbers, alphanumeric combos, and single chars (except 'a') as NOUN
    for token in doc:
        if token.text.isdigit() or re.match(r"^[A-Za-z]\d+[A-Za-z]$", token.text) or (len(token.text) == 1 and token.text.lower() != 'a'):
            token_pos = "NOUN"
        else:
            token_pos = token.pos_
        
        tagged_tokens.append((token.text, token_pos))

    # Filter out unwanted POS before counting total tokens
    filtered_tokens = [token for token, pos in tagged_tokens if pos not in EXCLUDE_POS]
    total_tokens = len(filtered_tokens)

    # Count POS occurrences, merging NOUN & PROPN as a single NOUN category
    pos_counts = Counter()
    for token, pos in tagged_tokens:
        if pos in {"NOUN", "PROPN"}:
            pos_counts["NOUN"] += 1  # Merge NOUN & PROPN
        else:
            pos_counts[pos] += 1  # Count other POS normally
    
    # Calculate noun percentage
    noun_count = pos_counts.get("NOUN", 0)
    noun_percentage = (noun_count / total_tokens) * 100 if total_tokens > 0 else 0
    
    # Calculate entropy
    entropy = pos_entropy(pos_counts, total_tokens) if total_tokens > 0 else 0

    # Classify text
    classification = "🔹 Keyword-heavy / Structured Query" if noun_percentage > 50 and entropy < 2.5 else "🔹 General Text"
    
    return noun_percentage, entropy, classification, pos_counts, tagged_tokens

# Streamlit App UI
st.title("📊 Text Analysis: Noun Percentage, POS Diversity & Token Tags")
st.markdown("Analyze how noun-heavy and diverse your input text is, and view POS tagging.")

# User Input
user_input = st.text_area("Enter your text:", "Buy the best Samsung phone with 8GB RAM and 120Hz display.")

if st.button("Analyze"):
    if user_input.strip():
        noun_pct, entropy, classification, pos_counts, tagged_tokens = analyze_text(user_input)
        
        # Display Results
        st.subheader("📝 Analysis Result:")
        st.write(f"*Noun Percentage:* {noun_pct:.2f}%")
        st.write(f"*POS Diversity (Entropy):* {entropy:.2f}")
        st.write(f"*Classification:* {classification}")
        
        # Display POS Breakdown
        st.subheader("📌 POS Tag Breakdown:")
        st.write(dict(pos_counts))
        
        # Display Token POS Tags
        st.subheader("🔍 Tokenized Words & POS Tags:")
        for token, pos in tagged_tokens:
            st.write(f"*{token}* → {pos}")

    else:
        st.warning("⚠️ Please enter some text for analysis.")

# Footer
st.markdown("---")
st.markdown("🔹 *Noun Percentage*: Measures how noun-heavy the text is.")
st.markdown("🔹 *POS Diversity (Entropy)*: Measures how varied the sentence structure is.")
st.markdown("🔹 *Classification*: Determines if the input is a keyword-heavy query, conversational text, or balanced.")
