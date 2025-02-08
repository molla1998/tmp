import streamlit as st
import stanza
import torch
import time
import numpy as np
from collections import Counter

# Initialize Stanza Pipeline
stanza.download("en")  # Download English model if not already present
nlp = stanza.Pipeline("en", use_gpu=torch.cuda.is_available())  # Load Stanza model

# Streamlit UI
st.title("📊 Text Analysis: Noun Percentage & POS Diversity (Stanza-powered)")
st.markdown("Analyze your text using **Stanza-based POS tagging** for accurate linguistic insights.")

# Select device (CPU or GPU)
device_option = st.radio("Select Inference Device:", ["CPU", "GPU"], index=0)
use_gpu = device_option == "GPU" and torch.cuda.is_available()

st.write(f"Using {'GPU' if use_gpu else 'CPU'} for inference.")

# Function to calculate POS entropy
def pos_entropy(pos_counts, total_tokens):
    probabilities = [count / total_tokens for count in pos_counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

# Function to analyze text using Stanza
def analyze_text(text):
    start_time = time.time()  # Start timing

    doc = nlp(text)  # Process text with Stanza
    total_tokens = sum(len(sentence.words) for sentence in doc.sentences)
    
    # Extract POS tags
    pos_tags = [word.upos for sentence in doc.sentences for word in sentence.words]
    pos_counts = Counter(pos_tags)

    noun_count = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
    noun_percentage = (noun_count / total_tokens) * 100 if total_tokens > 0 else 0
    entropy = pos_entropy(pos_counts, total_tokens) if total_tokens > 0 else 0

    # Classify text
    classification = "🔹 Keyword-heavy / Structured Query" if noun_percentage > 50 and entropy < 2.5 else "🔹 General Text"

    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    return noun_percentage, entropy, classification, pos_counts, inference_time

# User Input
user_input = st.text_area("Enter your text:", "Buy the best Samsung phone with 8GB RAM and 120Hz display.")

if st.button("Analyze"):
    if user_input.strip():
        noun_pct, entropy, classification, pos_counts, inference_time = analyze_text(user_input)
        
        # Display Results
        st.subheader("📝 Analysis Result:")
        st.write(f"**Noun Percentage:** {noun_pct:.2f}%")
        st.write(f"**POS Diversity (Entropy):** {entropy:.2f}")
        st.write(f"**Classification:** {classification}")
        st.write(f"⏱️ **Inference Time:** {inference_time:.2f} ms")

        # Display POS Breakdown
        st.subheader("📌 POS Tag Breakdown:")
        st.write(dict(pos_counts))
    else:
        st.warning("⚠️ Please enter some text for analysis.")

# Footer
st.markdown("---")
st.markdown("🔹 **Noun Percentage**: Measures how noun-heavy the text is.")
st.markdown("🔹 **POS Diversity (Entropy)**: Measures how varied the sentence structure is.")
st.markdown("🔹 **Classification**: Determines if the input is a keyword-heavy query or general text.")
st.markdown("🔹 **Inference Time**: Measures how long Stanza takes for processing.")
