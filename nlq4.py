import streamlit as st
import torch
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from collections import Counter

# Load model and tokenizer
MODEL_NAME = "vblagoje/bert-english-uncased-finetuned-pos"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

# Set up Streamlit UI
st.title("📊 Text Analysis: Noun Percentage & POS Diversity (BERT-powered)")
st.markdown("Analyze your text using **BERT-based POS tagging** for more accurate results.")

# Select device (CPU or GPU)
device_option = st.radio("Select Inference Device:", ["CPU", "GPU"], index=0)
device = torch.device("cuda" if (device_option == "GPU" and torch.cuda.is_available()) else "cpu")
model.to(device)

# POS Tagging Pipeline
nlp_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, device=0 if device.type == "cuda" else -1)

# Function to calculate POS entropy
def pos_entropy(pos_counts, total_tokens):
    probabilities = [count / total_tokens for count in pos_counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

# Function to analyze text using BERT POS tagging
def analyze_text(text):
    start_time = time.time()  # Start timing

    tokens = tokenizer.tokenize(text)
    predictions = nlp_pipeline(text)

    total_tokens = len(tokens)
    pos_counts = Counter([pred["entity"].split("-")[-1] for pred in predictions])  # Extract POS tags

    noun_count = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
    noun_percentage = (noun_count / total_tokens) * 100 if total_tokens > 0 else 0
    entropy = pos_entropy(pos_counts, total_tokens) if total_tokens > 0 else 0

    # Classify text
    if noun_percentage > 50 and entropy < 2.5:
        classification = "🔹 Keyword-heavy / Structured Query"
    else:
        classification = "🔹 General Text"

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
st.markdown("🔹 **Inference Time**: Measures how long BERT takes for processing.")

