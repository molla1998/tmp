import streamlit as st
from transformers import pipeline
import torch

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load the POS tagging model
@st.cache_resource
def load_model():
    return pipeline("token-classification", model="dbmdz/bert-large-cased-finetuned-conll03-english", device=device)

nlp = load_model()

# Function to generate plain text POS tagging output
def generate_pos_output(text):
    results = nlp(text)
    pos_output = []
    for result in results:
        word = result['word'].replace("##", "")  # Handle subword tokens
        pos_tag = result['entity_group']
        pos_output.append(f"{word} -> {pos_tag}")
    return "\n".join(pos_output)

# Streamlit App
st.title("State-of-the-Art POS Tagging (GPU Enabled)")
st.markdown("Enter text in the input box, and click **Submit** to see the tagged output.")

# Display whether GPU is being used
if torch.cuda.is_available():
    st.success("🚀 Using GPU for inference")
else:
    st.warning("⚠️ GPU not available, using CPU instead")

# Input Text Box
user_input = st.text_area("Enter your text here:", "The quick brown fox jumps over the lazy dog.")

# Submit Button
if st.button("Submit"):
    if user_input.strip():
        st.markdown("### POS Tagged Output:")
        try:
            pos_output = generate_pos_output(user_input)
            if pos_output.strip():
                st.text(pos_output)
            else:
                st.warning("No tags returned by the model. Try another input.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter some text before submitting!")
