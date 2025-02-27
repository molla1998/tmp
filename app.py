import streamlit as st
import spacy
from collections import Counter
import numpy as np
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define POS tags to exclude
EXCLUDE_POS = {"PUNCT", "SYM", "DET", "CCONJ", "PART", "SCONJ"}

# Function to calculate POS entropy
def pos_entropy(pos_counts, total_tokens):
    probabilities = [count / total_tokens for count in pos_counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

# Function to analyze text based on selected approach
def analyze_text(text, approach, noun_threshold, entropy_threshold):
    doc = nlp(text)
    
    # Filter out unwanted POS before counting total tokens
    filtered_tokens = [token for token in doc if token.pos_ not in EXCLUDE_POS]
    total_tokens = len(filtered_tokens)
    
    # Count POS only from meaningful tokens
    pos_counts = Counter([token.pos_ for token in filtered_tokens])
    
    noun_count = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
    noun_percentage = (noun_count / total_tokens) * 100 if total_tokens > 0 else 0
    entropy = pos_entropy(pos_counts, total_tokens) if total_tokens > 0 else 0
    
    # Classification based on selected approach
    if approach == "Approach 1":
        if noun_percentage <= noun_threshold and entropy >= entropy_threshold:
            classification = "🔹 Natural Language Query"
        else:
            classification = "🔹 Keyword-Heavy"
    elif approach == "Approach 2":
        if entropy >= entropy_threshold:
            classification = "🔹 Natural Language Query"
        else:
            classification = "🔹 Keyword-Heavy"
    
    return noun_percentage, entropy, classification, pos_counts

# Streamlit UI
st.title("📊 Text Analysis: Noun Percentage & POS Diversity")
st.markdown("Analyze how noun-heavy and diverse your input text is.")

# Dropdown for selecting approach
approach = st.selectbox("Select Classification Approach:", ["Approach 1", "Approach 2"])

# Sidebar for threshold configuration
st.sidebar.header("⚙️ Configure Thresholds")
noun_threshold = st.sidebar.slider("Noun Percentage Threshold", min_value=10, max_value=90, value=50)
entropy_threshold = st.sidebar.slider("Entropy Threshold", min_value=0.5, max_value=3.0, value=1.5 if approach == "Approach 1" else 2.0)

# User Input
user_input = st.text_area("Enter your text:", "Buy the best Samsung phone with 8GB RAM and 120Hz display.")

if st.button("Analyze"):
    if user_input.strip():
        noun_pct, entropy, classification, pos_counts = analyze_text(user_input, approach, noun_threshold, entropy_threshold)
        
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

# Comment Box for User Feedback
st.subheader("💬 User Reviews")
user_comment = st.text_area("Leave your feedback here:")

if st.button("Submit Review"):
    if user_comment.strip():
        with open("user_reviews.txt", "a") as f:
            f.write(user_comment + "\n---\n")
        st.success("✅ Your review has been saved!")
    else:
        st.warning("⚠️ Please enter a valid comment.")

# Footer
st.markdown("---")
st.markdown("🔹 **Noun Percentage**: Measures how noun-heavy the text is.")
st.markdown("🔹 **POS Diversity (Entropy)**: Measures how varied the sentence structure is.")
st.markdown("🔹 **Classification**: Determines if the input is a keyword-heavy query, conversational text, or balanced.")
