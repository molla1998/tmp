import streamlit as st
import torch
import time
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model on CPU
MODEL_PATH = "C:\\Users\\molla\\Downloads\\Samsung_task\\output\\nlq_classifier_fixed"
device = torch.device("cpu")
model = SentenceTransformer(MODEL_PATH, device=device)

# Classification weights/bias path (if available)
# Replace below with your actual weights if you saved them during training
# For now, we'll use a mock thresholding for demo purposes.

st.set_page_config(page_title="NLQ Classifier", layout="centered")
st.title("📱 Samsung NLQ Classifier")
st.markdown("Enter a Korean query below to classify it as **NLQ (1)** or **KH (0)**.")

query = st.text_input("🔍 Enter your query:")

# Dummy classification head
# Replace this logic with a real one if you saved classification weights
def classify(embedding):
    # This is a placeholder for a real classifier head.
    # For now, we simulate with a simple linear projection using cosine similarity.
    # You can replace this with an actual trained classifier head if saved.
    # For demo, we simulate NLQ vs KH by thresholding cosine similarity to a fixed prototype.
    kh_proto = torch.randn_like(embedding)
    sim = torch.nn.functional.cosine_similarity(embedding, kh_proto)
    return int(sim < 0.5)  # if similarity low, assume it's NLQ (1)

if query:
    with st.spinner("Classifying..."):
        start_time = time.time()

        # Encode
        embedding = model.encode([query], convert_to_tensor=True)

        # Fake classification (replace with real logic if you saved classifier head)
        prediction = classify(embedding)

        elapsed_ms = (time.time() - start_time) * 1000

    label_name = "NLQ (1)" if prediction == 1 else "KH (0)"
    st.success(f"✅ **Prediction:** {label_name}")
    st.info(f"⏱️ Inference time: `{elapsed_ms:.2f} ms`")

