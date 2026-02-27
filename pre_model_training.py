import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("dataset.csv")

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# ==============================
# TOKENIZER
# ==============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Add special tokens if not present
special_tokens = {"additional_special_tokens": ["[P1]", "[P2]"]}
tokenizer.add_special_tokens(special_tokens)

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

        masked_query = row["masked_query"]
        product_1 = row["product_1"]
        product_2 = row["product_2"]
        label = int(row["label"])

        # Combine input
        encoding = tokenizer(
            masked_query,
            product_1,
            product_2,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

        return item

# ==============================
# DATALOADERS
# ==============================
train_dataset = RelationDataset(train_df)
val_dataset = RelationDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ==============================
# MODEL
# ==============================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

model.resize_token_embeddings(len(tokenizer))
model.to(DEVICE)

# ==============================
# OPTIMIZER + SCHEDULER
# ==============================
optimizer = AdamW(model.parameters(), lr=LR)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# ==============================
# TRAINING LOOP
# ==============================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # ==============================
    # VALIDATION
    # ==============================
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average="macro")

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Accuracy: {acc:.4f}")
    print(f"Val F1: {f1:.4f}")

# ==============================
# SAVE MODEL
# ==============================
model.save_pretrained("relation_model")
tokenizer.save_pretrained("relation_model")

print("Training complete. Model saved.")
