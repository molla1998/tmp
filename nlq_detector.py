import streamlit as st
import spacy
from collections import Counter

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to classify query
def classify_nl_query(text):
    doc = nlp(text)
    
    # POS Tagging: Count important POS categories
    pos_counts = Counter([token.pos_ for token in doc])
    total_tokens = len(doc)
    
    # Compute Scores
    pos_score = ((pos_counts['NOUN'] + pos_counts['VERB'] + pos_counts['ADJ'] + pos_counts['ADV']) / total_tokens) * 100 if total_tokens else 0
    ner_score = (len(doc.ents) / total_tokens) * 100 if total_tokens else 0
    stopword_score = (sum(1 for token in doc if token.is_stop) / total_tokens) * 100 if total_tokens else 0

    # Final weighted score
    final_score = (0.5 * pos_score) + (0.3 * ner_score) + (0.2 * stopword_score)

    # Classification Threshold
    is_natural = final_score > 50

    return {
        "nlp_query": "✅ Yes" if is_natural else "❌ No",
        "POS Score": f"{pos_score:.2f}%",
        "NER Score": f"{ner_score:.2f}%",
        "Stopword Score": f"{stopword_score:.2f}%",
        "Final Score": f"{final_score:.2f}%"
    }

# Streamlit App
st.title("🔍 Natural Language Query Detector")
st.markdown("This app analyzes your input and determines whether it is a **natural language query** or not.")

# Input Box
user_input = st.text_area("Enter your query:", "Show me the best smartphone under 10k with 8GB RAM")

# Submit Button
if st.button("Analyze"):
    if user_input.strip():
        result = classify_nl_query(user_input)

        # Display Results
        st.subheader("📝 Classification Result:")
        st.write(f"**Natural Language Query?** {result['nlp_query']}")
        
        st.subheader("📊 Scores:")
        st.write(f"- **POS Score:** {result['POS Score']}")
        st.write(f"- **NER Score:** {result['NER Score']}")
        st.write(f"- **Stopword Score:** {result['Stopword Score']}")
        st.write(f"- **Final Score:** {result['Final Score']}")
    else:
        st.warning("⚠️ Please enter some text for analysis.")

# Footer
st.markdown("---")
st.markdown("🔹 **POS Score**: Measures the proportion of meaningful words (nouns, verbs, adjectives).")
st.markdown("🔹 **NER Score**: Checks how many words are named entities (e.g., products, places).")
st.markdown("🔹 **Stopword Score**: Evaluates common words like 'the', 'is', 'to' which indicate natural language.")
st.markdown("🔹 **Final Score**: Combined metric to classify the input.")
