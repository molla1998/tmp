import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm
import numpy as np

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "relation_model"
TEST_FILE = "testdataset.csv"
BATCH_SIZE = 32
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# LOAD MODEL + TOKENIZER
# ==============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# ==============================
# LOAD TEST DATA
# ==============================
df = pd.read_csv(TEST_FILE)

# ==============================
# DATASET CLASS
# ==============================
class RelationDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        encoding = tokenizer(
            row["masked_query"],  # change if using full query
            row["product_1"],
            row["product_2"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(int(row["label"]), dtype=torch.long)
        }

# ==============================
# DATALOADER
# ==============================
test_dataset = RelationDataset(df)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ==============================
# INFERENCE
# ==============================
all_preds = []
all_labels = []
all_confidences = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Running Inference"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(probs.max(dim=1).values.cpu().numpy())

# ==============================
# METRICS
# ==============================
accuracy = accuracy_score(all_labels, all_preds)
macro_precision = precision_score(all_labels, all_preds, average="macro")
macro_recall = recall_score(all_labels, all_preds, average="macro")
macro_f1 = f1_score(all_labels, all_preds, average="macro")

print("\n==============================")
print("       Evaluation Results")
print("==============================")
print(f"Accuracy        : {accuracy:.4f}")
print(f"Macro Precision : {macro_precision:.4f}")
print(f"Macro Recall    : {macro_recall:.4f}")
print(f"Macro F1        : {macro_f1:.4f}")

print("\nDetailed Classification Report:\n")
print(classification_report(all_labels, all_preds, digits=4))

print("\nConfusion Matrix:\n")
print(confusion_matrix(all_labels, all_preds))

# ==============================
# SAVE PREDICTIONS
# ==============================
df["predicted_label"] = all_preds
df["confidence"] = np.round(all_confidences, 4)

df.to_csv("test_predictions.csv", index=False)

print("\nPredictions saved to test_predictions.csv")
