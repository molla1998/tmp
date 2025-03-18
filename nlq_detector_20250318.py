import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time

def load_model():
    model_name = "your-finetuned-model"  # Replace with your Hugging Face model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def predict(query, tokenizer, model):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end_time = time.time()
    
    pred = torch.argmax(outputs.logits).item()
    inference_time = round((end_time - start_time) * 1000, 2)  # in ms
    
    return "NLQ" if pred == 1 else "KH", inference_time

def post_processing(query):
    """Placeholder function for post-processing."""
    return "NLQ" if "example" in query.lower() else "KH"

# Load model and tokenizer
st.title("🧠 Natural Language Query Detector 🔍")
st.write("### 📌 Definitions:")
st.write("- 🔵 **KH:** Keyword Heavy Query")
st.write("- 🟢 **NLQ:** Natural Language Query")

tokenizer, model = load_model()

# User Input Section
st.subheader("💬 Enter Your Query")
query = st.text_area("✍️ Type your query here:")

if st.button("🚀 Predict"):
    if query.strip():
        pred_result, inf_time = predict(query, tokenizer, model)
        st.session_state.pred_result = pred_result  # Save result in session
        st.session_state.inf_time = inf_time
        st.success(f"🎯 **Prediction:** {pred_result}")
        st.info(f"⏱️ **Inference Time:** {inf_time} ms")
    else:
        st.warning("⚠️ Please enter a query!")

# Post Processing Button (Enabled only if first result is KH)
if "pred_result" in st.session_state and st.session_state.pred_result == "KH":
    if st.button("🔄 Post Processing"):
        post_result = post_processing(query)
        st.success(f"✅ **Post Processing Result:** {post_result}")

# User Comments Section
st.subheader("📝 User Reviews / Comments")
user_comment = st.text_area("💡 Share your thoughts:")
if st.button("💾 Save Review"):
    with open("user_reviews.txt", "a") as f:
        f.write(user_comment + "\n")
    st.success("📌 Review Saved!")
