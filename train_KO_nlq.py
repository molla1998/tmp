import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, LoggingHandler, models
from torch.utils.data import DataLoader
import logging
import os

# Logging setup
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, handlers=[LoggingHandler()])

# Config
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_PATH = "korean_training_dataset_20250406-0240.csv"
OUTPUT_PATH = "output/nlq_classifier_fixed"
BATCH_SIZE = 32
EPOCHS = 30

# Load data
df = pd.read_csv(DATA_PATH)
df['label'] = df['label'].map({'KH': 0, 'NLQ': 1})
df = df.sample(frac=1).reset_index(drop=True)

# Split
train_ratio = 0.8
split = int(len(df) * train_ratio)
train_df = df.iloc[:split]
val_df = df.iloc[split:]

# InputExample format
train_samples = [InputExample(texts=[row['query'], row['query']], label=int(row['label'])) for _, row in train_df.iterrows()]
val_samples = [InputExample(texts=[row['query'], row['query']], label=int(row['label'])) for _, row in val_df.iterrows()]


# Model and loss
model = SentenceTransformer(MODEL_NAME)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)

loss = losses.SoftmaxLoss(
    model=model,
    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    num_labels=2
)

# Optional evaluator (commented out, can add later)
# from sentence_transformers.evaluation import LabelAccuracyEvaluator
# val_texts = [x.texts[0] for x in val_samples]
# val_labels = [int(x.label) for x in val_samples]
# evaluator = LabelAccuracyEvaluator(val_texts, val_labels)

# Fine-tune the model
model.fit(
    train_objectives=[(train_dataloader, loss)],
    epochs=EPOCHS,
    output_path=OUTPUT_PATH,
    save_best_model=True,
    use_amp=True  # Mixed precision (faster on GPU)
)

print(f"✅ Model saved to: {OUTPUT_PATH}")
