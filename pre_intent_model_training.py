import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from tqdm import tqdm
import numpy as np

# ==============================
# CONFIG
# ==============================
MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"
BATCH_SIZE = 16
EPOCHS = 5
MAX_LEN = 128
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("dataset.csv")

# Intent Mapping
intent_map = {
    "browse": 0,
    "transactional": 1,
    "comparison": 2,
    "information": 3
}

df["intent_label"] = df["category"].map(intent_map)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# ==============================
# DATASET CLASS
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
            max_length=MAX_LEN,
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
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        relation_logits = self.relation_classifier(cls_output)
        intent_logits = self.intent_classifier(cls_output)
        
        return relation_logits, intent_logits

# ==============================
# INIT
# ==============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = MultiTaskModel(MODEL_NAME)
model.to(DEVICE)

train_dataset = MultiTaskDataset(train_df, tokenizer)
val_dataset = MultiTaskDataset(val_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

relation_loss_fn = nn.CrossEntropyLoss()
intent_loss_fn = nn.CrossEntropyLoss()

best_precision = 0.0

# ==============================
# TRAINING LOOP
# ==============================
for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        relation_labels = batch["relation_label"].to(DEVICE)
        intent_labels = batch["intent_label"].to(DEVICE)

        relation_logits, intent_logits = model(input_ids, attention_mask)

        relation_loss = relation_loss_fn(relation_logits, relation_labels)
        intent_loss = intent_loss_fn(intent_logits, intent_labels)

        loss = relation_loss + intent_loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # ==============================
    # VALIDATION
    # ==============================
    model.eval()
    relation_preds = []
    relation_true = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            relation_labels = batch["relation_label"].to(DEVICE)

            relation_logits, _ = model(input_ids, attention_mask)
            preds = torch.argmax(relation_logits, dim=1)

            relation_preds.extend(preds.cpu().numpy())
            relation_true.extend(relation_labels.cpu().numpy())

    macro_precision = precision_score(relation_true, relation_preds, average="macro")

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Validation Macro Precision (Relation): {macro_precision:.4f}")

    # ==============================
    # SAVE BEST MODEL
    # ==============================
    if macro_precision > best_precision:
        best_precision = macro_precision
        print("Saving Best Model...\n")

        torch.save(model.state_dict(), "best_multitask_model.pt")
        tokenizer.save_pretrained("best_multitask_tokenizer")

print("Training Complete.")
