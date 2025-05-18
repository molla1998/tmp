import pandas as pd
import mlflow
import mlflow.pytorch
from sentence_transformers import SentenceTransformer, InputExample, losses, LoggingHandler, models
from torch.utils.data import DataLoader
import logging
import os
import shutil

mlflow.set_tracking_uri("http://localhost:5000")

# Logging setup
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, handlers=[LoggingHandler()])

# Config
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_PATH = "sample_dataset.csv"
OUTPUT_PATH = "output/nlq_classifier_fixed"
BATCH_SIZE = 32
EPOCHS = 10

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

# Start MLflow run
with mlflow.start_run(run_name="sentence_transformers_binary_classifier"):

    # Log config/hyperparameters
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("train_samples", len(train_samples))
    mlflow.log_param("val_samples", len(val_samples))

    # Model and loss
    model = SentenceTransformer(MODEL_NAME)
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)

    loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=2
    )

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=EPOCHS,
        output_path=OUTPUT_PATH,
        save_best_model=True,
        use_amp=True
    )

    print(f"✅ Model saved to: {OUTPUT_PATH}")

    # Log model directory as an MLflow artifact
    mlflow.log_artifacts(OUTPUT_PATH, artifact_path="model")

    # Optional: register the model in the MLflow Model Registry
    mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "NLQClassifier")
