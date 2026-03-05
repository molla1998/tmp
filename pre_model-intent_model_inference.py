import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "best_multitask_model.pt"
TOKENIZER_PATH = "best_multitask_tokenizer"
TEST_FILE = "testdataset.csv"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# INTENT MAP
# ==============================
intent_map = {
    "browse": 0,
    "transactional": 1,
    "comparison": 2,
    "information": 3
}

# ==============================
# DATASET
# ==============================
class MultiTaskDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        encoding = self.tokenizer(
            row["query"],
            row["product_1"],
            row["product_2"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "relation_label": torch.tensor(int(row["relation_label"]), dtype=torch.long),
            "intent_label": torch.tensor(int(row["intent_label"]), dtype=torch.long)
        }

# ==============================
# MODEL
# ==============================
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_relation_labels=3, num_intent_labels=4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.relation_classifier = nn.Linear(hidden_size, num_relation_labels)
        self.intent_classifier = nn.Linear(hidden_size, num_intent_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        relation_logits = self.relation_classifier(cls_output)
        intent_logits = self.intent_classifier(cls_output)
        return relation_logits, intent_logits


# ==============================
# LOAD MODEL
# ==============================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = MultiTaskModel("microsoft/MiniLM-L12-H384-uncased")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(TEST_FILE)
df["intent_label"] = df["category"].map(intent_map)

dataset = MultiTaskDataset(df, tokenizer)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# ==============================
# EVALUATION
# ==============================
relation_preds = []
relation_true = []
intent_preds = []
intent_true = []

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        relation_labels = batch["relation_label"].numpy()
        intent_labels = batch["intent_label"].numpy()

        relation_logits, intent_logits = model(input_ids, attention_mask)

        relation_pred = torch.argmax(relation_logits, dim=1).cpu().numpy()
        intent_pred = torch.argmax(intent_logits, dim=1).cpu().numpy()

        relation_preds.extend(relation_pred)
        relation_true.extend(relation_labels)

        intent_preds.extend(intent_pred)
        intent_true.extend(intent_labels)


# ==============================
# METRICS
# ==============================
print("\n==== RELATION METRICS ====")
print("Precision:", precision_score(relation_true, relation_preds, average="macro"))
print("Recall:", recall_score(relation_true, relation_preds, average="macro"))
print("F1:", f1_score(relation_true, relation_preds, average="macro"))
print("\nClassification Report:\n")
print(classification_report(relation_true, relation_preds))

print("\n==== INTENT METRICS ====")
print("Precision:", precision_score(intent_true, intent_preds, average="macro"))
print("Recall:", recall_score(intent_true, intent_preds, average="macro"))
print("F1:", f1_score(intent_true, intent_preds, average="macro"))
print("\nClassification Report:\n")
print(classification_report(intent_true, intent_preds))
