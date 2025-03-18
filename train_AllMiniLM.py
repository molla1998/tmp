import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=0.2)
    return train_texts, train_labels, val_texts, val_labels

def preprocess_data(texts, labels, tokenizer):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    return Dataset.from_dict({"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"], "labels": labels})

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", num_labels=2)

# Load and preprocess data
train_texts, train_labels, val_texts, val_labels = load_data("data.csv")
train_dataset = preprocess_data(train_texts, train_labels, tokenizer)
val_dataset = preprocess_data(val_texts, val_labels, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the best model continuously based on highest accuracy
trainer.save_model("./best_nlq_model")
