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

# Function to generate color-coded output
def generate_pos_highlighted_text(text):
    results = nlp(text)
    highlighted_text = ""
    for result in results:
        word = result['word']
        pos_tag = result['entity_group']
        # Assigning a transparent color style based on POS tags
        color = {
            "VERB": "rgba(255, 0, 0, 0.2)",  # Light red for verbs
            "NOUN": "rgba(0, 255, 0, 0.2)",  # Light green for nouns
            "ADJ": "rgba(0, 0, 255, 0.2)",  # Light blue for adjectives
            "ADP": "rgba(255, 165, 0, 0.2)",  # Light orange for prepositions
            "DET": "rgba(128, 0, 128, 0.2)",  # Light purple for determiners
        }.get(pos_tag, "rgba(211, 211, 211, 0.2)")  # Default light gray

        highlighted_text += f'<span style="background-color: {color}; padding: 4px; margin: 2px;">{word}</span> '
    
    return highlighted_text.strip()

# Streamlit App
st.title("State-of-the-Art POS Tagging (GPU Enabled)")
st.markdown("Enter text in the input box, and you'll see the text color-coded based on POS tags.")

# Display whether GPU is being used
if torch.cuda.is_available():
    st.success("🚀 Using GPU for inference")
else:
    st.warning("⚠️ GPU not available, using CPU instead")

# Input Text Box
user_input = st.text_area("Enter your text here:", "The quick brown fox jumps over the lazy dog.")

# Output Section
if user_input:
    st.markdown("### POS Tagged Output:")
    highlighted_output = generate_pos_highlighted_text(user_input)
    st.markdown(highlighted_output, unsafe_allow_html=True)
