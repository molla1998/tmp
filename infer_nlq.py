import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, confusion_matrix

# Load dataset (Assuming CSV has columns: 'query', 'noun_percentage', 'entropy', 'true_label')
df = pd.read_csv("test_queries.csv")

# Ensure "KH" is mapped to True and "NLQ" to False
df["true_label"] = df["true_label"].map({"KH": True, "NLQ": False})

# Get user-defined thresholds
noun_thr = float(input("Enter Noun Percentage Threshold: "))  # Example: 50
entropy_thr = float(input("Enter Entropy Threshold: "))  # Example: 2.0

# Predict labels using given thresholds
df["predicted_label"] = (df["noun_percentage"] > noun_thr) & (df["entropy"] < entropy_thr)

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(df["true_label"], df["predicted_label"]).ravel()

# Compute precision
precision = precision_score(df["true_label"], df["predicted_label"])

# Display results
print("\n🔹 Results 🔹")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Negatives (TN): {tn}")
print(f"Precision: {precision:.4f}")

# Save predictions with correct mapping
df.to_csv("predicted_queries.csv", index=False)
