import streamlit as st
import stanza
import torch

# Load the Stanza model for English
@st.cache_resource
def load_model():
    # Initialize the Stanza pipeline for POS tagging
    return stanza.Pipeline('en', processors='tokenize,mwt,pos')

nlp = load_model()

# Function to generate plain text POS tagging output
def generate_pos_output(text):
    doc = nlp(text)
    pos_output = []
    for sentence in doc.sentences:
        for word in sentence.words:
            pos_output.append(f"{word.text} -> {word.pos}")
    return "\n".join(pos_output)

# Streamlit App
st.title("State-of-the-Art POS Tagging with Stanza")
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
