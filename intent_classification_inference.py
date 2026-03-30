import pandas as pd
import torch
import numpy as np
import pickle

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix
)

# -----------------------
# Load model + tokenizer
# -----------------------
model_path = "./best_intent_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

model.eval()

# -----------------------
# Load label encoder
# -----------------------
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -----------------------
# Prediction function
# -----------------------
def predict_single(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)

    pred_id = torch.argmax(probs, dim=1).item()
    pred_label = le.inverse_transform([pred_id])[0]

    return {
        "label": pred_label,
        "confidence": probs[0][pred_id].item()
    }

# -----------------------
# Batch evaluation
# -----------------------
def evaluate_csv(csv_path):
    df = pd.read_csv(csv_path)  # columns: query, intent

    texts = df["query"].tolist()
    true_labels = df["intent"].tolist()

    true_ids = le.transform(true_labels)

    preds = []

    for text in texts:
        pred = predict_single(text)
        pred_id = le.transform([pred["label"]])[0]
        preds.append(pred_id)

    preds = np.array(preds)

    # -----------------------
    # Metrics
    # -----------------------
    acc = accuracy_score(true_ids, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_ids,
        preds,
        average="weighted"
    )

    print("\n===== Overall Metrics =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # -----------------------
    # Per-class report
    # -----------------------
    print("\n===== Classification Report =====")
    print(classification_report(
        true_ids,
        preds,
        target_names=le.classes_
    ))

    # -----------------------
    # Confusion Matrix
    # -----------------------
    print("\n===== Confusion Matrix =====")
    print(confusion_matrix(true_ids, preds))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    # Evaluate CSV
    evaluate_csv("test.csv")

    # Single inference
    print("\n===== Single Prediction =====")
    print(predict_single("बुक a flight ticket"))
