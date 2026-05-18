# multitask_distilbert_training_pipeline.py

```python
import ast
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score as ner_f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoModel, AutoTokenizer

############################################################
# CONFIG
############################################################
CONFIG = {
    "model_name": "distilbert-base-uncased",
    "max_length": 64,
    "batch_size": 8,
    "lr": 2e-5,
    "epochs": 5,
    "loss_alpha": 0.5,  # intent loss weight
    "loss_beta": 0.5,   # main-product loss weight
    "random_seed": 42,
    "test_size": 0.1,
    "csv_path": "data.csv",
    "save_dir": "saved_models"
}

############################################################
# LABEL CONFIG
############################################################
IGNORE_LABELS = {
    "RANDOM_NUMBER"
}

MAIN_PRODUCT_LABELS = {
    "PRODUCT_NAME",
    "ACCESSORY"
}

############################################################
# SEED
############################################################
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["random_seed"])

############################################################
# LOAD CSV
############################################################
print("Loading dataset...")

df = pd.read_csv(CONFIG["csv_path"])

print("Total samples:", len(df))

############################################################
# TRAIN / VAL SPLIT
############################################################
train_df, val_df = train_test_split(
    df,
    test_size=CONFIG["test_size"],
    random_state=CONFIG["random_seed"],
    stratify=df["intent"],
    shuffle=True
)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))

############################################################
# ENTITY PARSER
############################################################
def parse_entities(entity_str):
    """
    Parses CSV entity column.

    Expected format:

    {
        'entities': [
            {
                'start': 0,
                'end': 11,
                'entity': 'product_name',
                'is_main_product': True
            }
        ]
    }
    """

    data = ast.literal_eval(entity_str)

    parsed = []

    for ent in data.get("entities", []):

        label = ent["entity"].upper()

        if label in IGNORE_LABELS:
            continue

        parsed.append({
            "start": int(ent["start"]),
            "end": int(ent["end"]),
            "label": label,
            "is_main": bool(ent.get("is_main_product", False))
        })

    return parsed

############################################################
# CONVERT DATAFRAME -> INTERNAL FORMAT
############################################################
def build_dataset(df):

    dataset = []

    for _, row in df.iterrows():

        dataset.append({
            "text": row["query"],
            "entities": parse_entities(row["entities"]),
            "intent": row["intent"]
        })

    return dataset


train_data = build_dataset(train_df)
val_data = build_dataset(val_df)

############################################################
# BUILD LABELS
############################################################
def build_labels(data):

    ner_set = set()
    intent_set = set()

    for item in data:

        intent_set.add(item["intent"])

        for ent in item["entities"]:
            ner_set.add(ent["label"])

    ner_labels = ["O"]

    for label in sorted(ner_set):
        ner_labels.extend([
            f"B-{label}",
            f"I-{label}"
        ])

    intent_labels = sorted(list(intent_set))

    return ner_labels, intent_labels


ner_labels, intent_labels = build_labels(train_data)

print("NER Labels:")
print(ner_labels)

print("Intent Labels:")
print(intent_labels)

############################################################
# LABEL MAPS
############################################################
label2id = {
    l: i for i, l in enumerate(ner_labels)
}

id2label = {
    i: l for l, i in label2id.items()
}

intent2id = {
    l: i for i, l in enumerate(intent_labels)
}

id2intent = {
    i: l for l, i in intent2id.items()
}

############################################################
# TOKENIZER
############################################################
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["model_name"]
)

############################################################
# TOKEN ALIGNMENT
############################################################
def align_labels(text, entities, offsets):

    ner_tags = ["O"] * len(offsets)

    # -100 ignored in loss
    main_tags = [-100] * len(offsets)

    for ent in entities:

        start = ent["start"]
        end = ent["end"]
        label = ent["label"]

        for i, (s, e) in enumerate(offsets):

            # special tokens
            if s == e:
                continue

            if s >= start and e <= end:

                if s == start:
                    ner_tags[i] = f"B-{label}"
                else:
                    ner_tags[i] = f"I-{label}"

                # main-product labels only
                # for PRODUCT_NAME / ACCESSORY
                if label in MAIN_PRODUCT_LABELS:
                    main_tags[i] = 1 if ent["is_main"] else 0

    ner_ids = [
        label2id[tag]
        for tag in ner_tags
    ]

    return ner_ids, main_tags

############################################################
# DATASET CLASS
############################################################
class MultiTaskDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        encoding = tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=CONFIG["max_length"],
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offsets = encoding.pop("offset_mapping")[0]

        ner_ids, main_ids = align_labels(
            item["text"],
            item["entities"],
            offsets
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "ner_labels": torch.tensor(ner_ids),
            "main_labels": torch.tensor(main_ids),
            "intent_label": torch.tensor(
                intent2id[item["intent"]]
            )
        }

############################################################
# DATALOADERS
############################################################
train_dataset = MultiTaskDataset(train_data)
val_dataset = MultiTaskDataset(val_data)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False
)

############################################################
# MODEL
############################################################
class MultiTaskModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            CONFIG["model_name"]
        )

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)

        # NER HEAD
        self.ner_head = nn.Linear(
            hidden_size,
            len(ner_labels)
        )

        # INTENT HEAD
        self.intent_head = nn.Linear(
            hidden_size,
            len(intent_labels)
        )

        # MAIN PRODUCT HEAD
        self.main_head = nn.Linear(
            hidden_size,
            2
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        seq_output = self.dropout(
            outputs.last_hidden_state
        )

        cls_output = self.dropout(
            seq_output[:, 0]
        )

        ner_logits = self.ner_head(seq_output)

        intent_logits = self.intent_head(cls_output)

        main_logits = self.main_head(seq_output)

        return (
            ner_logits,
            intent_logits,
            main_logits
        )

############################################################
# DEVICE
############################################################
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

############################################################
# INIT MODEL
############################################################
model = MultiTaskModel().to(device)

############################################################
# OPTIMIZER
############################################################
optimizer = AdamW(
    model.parameters(),
    lr=CONFIG["lr"]
)

############################################################
# LOSSES
############################################################
ner_loss_fn = nn.CrossEntropyLoss()

intent_loss_fn = nn.CrossEntropyLoss()

main_loss_fn = nn.CrossEntropyLoss(
    ignore_index=-100
)

############################################################
# EVALUATION
############################################################
def evaluate(model, loader):

    model.eval()

    all_ner_preds = []
    all_ner_labels = []

    all_intent_preds = []
    all_intent_labels = []

    all_main_preds = []
    all_main_labels = []

    with torch.no_grad():

        for batch in loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            ner_labels_batch = batch["ner_labels"].cpu().numpy()
            main_labels_batch = batch["main_labels"].cpu().numpy()
            intent_labels_batch = batch["intent_label"].cpu().numpy()

            ner_logits, intent_logits, main_logits = model(
                input_ids,
                attention_mask
            )

            ner_preds = torch.argmax(
                ner_logits,
                dim=-1
            ).cpu().numpy()

            intent_preds = torch.argmax(
                intent_logits,
                dim=-1
            ).cpu().numpy()

            main_preds = torch.argmax(
                main_logits,
                dim=-1
            ).cpu().numpy()

            ####################################################
            # NER
            ####################################################
            for pred_seq, label_seq in zip(
                ner_preds,
                ner_labels_batch
            ):

                pred_tags = []
                label_tags = []

                for p, l in zip(pred_seq, label_seq):

                    pred_tags.append(id2label[p])
                    label_tags.append(id2label[l])

                all_ner_preds.append(pred_tags)
                all_ner_labels.append(label_tags)

            ####################################################
            # INTENT
            ####################################################
            all_intent_preds.extend(intent_preds)
            all_intent_labels.extend(intent_labels_batch)

            ####################################################
            # MAIN PRODUCT
            ####################################################
            for pred_seq, label_seq in zip(
                main_preds,
                main_labels_batch
            ):

                for p, l in zip(pred_seq, label_seq):

                    if l != -100:
                        all_main_preds.append(p)
                        all_main_labels.append(l)

    ner_f1 = ner_f1_score(
        all_ner_labels,
        all_ner_preds
    )

    intent_acc = accuracy_score(
        all_intent_labels,
        all_intent_preds
    )

    if len(all_main_labels) > 0:
        main_f1 = f1_score(
            all_main_labels,
            all_main_preds
        )
    else:
        main_f1 = 0.0

    return ner_f1, intent_acc, main_f1

############################################################
# TRAIN LOOP
############################################################
def train():

    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    best_score = 0

    for epoch in range(CONFIG["epochs"]):

        model.train()

        total_loss = 0

        for batch in train_loader:

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            ner_labels_batch = batch["ner_labels"].to(device)
            main_labels_batch = batch["main_labels"].to(device)
            intent_labels_batch = batch["intent_label"].to(device)

            ner_logits, intent_logits, main_logits = model(
                input_ids,
                attention_mask
            )

            ####################################################
            # NER LOSS
            ####################################################
            ner_loss = ner_loss_fn(
                ner_logits.view(
                    -1,
                    ner_logits.shape[-1]
                ),
                ner_labels_batch.view(-1)
            )

            ####################################################
            # INTENT LOSS
            ####################################################
            intent_loss = intent_loss_fn(
                intent_logits,
                intent_labels_batch
            )

            ####################################################
            # MAIN PRODUCT LOSS
            ####################################################
            main_loss = main_loss_fn(
                main_logits.view(-1, 2),
                main_labels_batch.view(-1)
            )

            ####################################################
            # COMBINED LOSS
            ####################################################
            loss = (
                ner_loss
                + CONFIG["loss_alpha"] * intent_loss
                + CONFIG["loss_beta"] * main_loss
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        ########################################################
        # VALIDATION
        ########################################################
        ner_f1, intent_acc, main_f1 = evaluate(
            model,
            val_loader
        )

        combined_score = (
            ner_f1
            + 0.5 * intent_acc
            + 0.5 * main_f1
        )

        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1}")
        print("=" * 60)

        print(f"Train Loss     : {total_loss:.4f}")
        print(f"NER F1         : {ner_f1:.4f}")
        print(f"Intent Acc     : {intent_acc:.4f}")
        print(f"Main Product F1: {main_f1:.4f}")
        print(f"Combined Score : {combined_score:.4f}")

        ########################################################
        # SAVE BEST MODEL
        ########################################################
        if combined_score > best_score:

            best_score = combined_score

            save_path = os.path.join(
                CONFIG["save_dir"],
                "best_model.pt"
            )

            torch.save({
                "model_state_dict": model.state_dict(),
                "label2id": label2id,
                "id2label": id2label,
                "intent2id": intent2id,
                "id2intent": id2intent,
                "config": CONFIG
            }, save_path)

            print("✅ Best model saved")

############################################################
# TRAIN
############################################################
train()

############################################################
# LOAD BEST MODEL
############################################################
def load_best_model(path):

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return checkpoint

############################################################
# INFERENCE
############################################################
def predict(text):

    model.eval()

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=CONFIG["max_length"],
        return_offsets_mapping=True
    )

    offsets = encoding.pop("offset_mapping")[0]

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():

        ner_logits, intent_logits, main_logits = model(
            input_ids,
            attention_mask
        )

    ########################################################
    # PREDICTIONS
    ########################################################
    ner_preds = torch.argmax(
        ner_logits,
        dim=-1
    )[0].cpu().numpy()

    intent_pred = torch.argmax(
        intent_logits,
        dim=-1
    ).item()

    main_preds = torch.argmax(
        main_logits,
        dim=-1
    )[0].cpu().numpy()

    ########################################################
    # TOKENS
    ########################################################
    tokens = tokenizer.convert_ids_to_tokens(
        input_ids[0]
    )

    results = []

    for token, ner_id, main_id, offset in zip(
        tokens,
        ner_preds,
        main_preds,
        offsets
    ):

        if offset[0] == offset[1]:
            continue

        results.append({
            "token": token,
            "ner": id2label[ner_id],
            "is_main_product": bool(main_id)
        })

    return {
        "intent": id2intent[intent_pred],
        "tokens": results
    }

############################################################
# EXAMPLE INFERENCE
############################################################
print("\nRunning sample inference...\n")

sample = predict(
    "show me iphone with charger"
)

print(sample)

```

# Run

```bash
python multitask_distilbert_training_pipeline.py
```

# Expected CSV Format

```csv
query,entities,intent
"iphone with charger","{'entities':[{'start':0,'end':7,'entity':'product_name','is_main_product':True},{'start':13,'end':20,'entity':'accessory','is_main_product':False}]}","product_search"
```
