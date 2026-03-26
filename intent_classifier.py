import pandas as pd
import torch
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

# -----------------------
# Step 1: Load CSV
# -----------------------
df = pd.read_csv("data.csv")  # columns: query, intent

texts = df["query"].tolist()
labels = df["intent"].tolist()

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# -----------------------
# Step 2: Compute class weights
# -----------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_encoded),
    y=labels_encoded
)

class_weights = torch.tensor(class_weights, dtype=torch.float)

# -----------------------
# Step 3: Create dataset
# -----------------------
dataset = Dataset.from_dict({
    "text": texts,
    "label": labels_encoded
})

dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label")

# -----------------------
# Step 4: Load MiniLM
# -----------------------
model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(le.classes_)
)

# -----------------------
# Step 5: Tokenization
# -----------------------
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize)

# -----------------------
# Step 6: Custom Trainer with class weights
# -----------------------
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

# -----------------------
# Step 7: Metrics (F1)
# -----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    f1 = f1_score(labels, preds, average="weighted")
    return {"f1": f1}

# -----------------------
# Step 8: Training args
# -----------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=5,

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    logging_dir="./logs",
    save_total_limit=2,
)

# -----------------------
# Step 9: Train
# -----------------------
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# -----------------------
# Step 10: Save best model
# -----------------------
trainer.save_model("./best_intent_model")
tokenizer.save_pretrained("./best_intent_model")

# Save label encoder
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
