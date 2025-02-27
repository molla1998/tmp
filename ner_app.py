import streamlit as st
import spacy
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit UI
st.title("📝 Named Entity Recognition (NER) App")
st.markdown("Extract and display named entities from text using **spaCy**.")

# User Input
user_input = st.text_area("Enter your text:", "Apple was founded by Steve Jobs in California.")

if st.button("Analyze"):
    if user_input.strip():
        # Process text with spaCy
        doc = nlp(user_input)
        
        # Extract Entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Display Results
        st.subheader("📌 Detected Entities:")
        if entities:
            for entity, label in entities:
                st.write(f"**{entity}** → `{label}`")
        else:
            st.write("No named entities detected.")

    else:
        st.warning("⚠️ Please enter some text for analysis.")

# Comment Box for User Feedback
st.subheader("💬 User Reviews")
user_comment = st.text_area("Leave your feedback here:")

if st.button("Submit Review"):
    if user_comment.strip():
        with open("ner_reviews.txt", "a") as f:
            f.write(user_comment + "\n---\n")
        st.success("✅ Your review has been saved!")
    else:
        st.warning("⚠️ Please enter a valid comment.")

# Footer
st.markdown("---")
st.markdown("🔹 **NER (Named Entity Recognition)** extracts entities like names, dates, locations, etc.")
st.markdown("🔹 **Model Used:** `en_core_web_sm` (spaCy's small English model)")
