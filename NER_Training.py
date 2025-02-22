import spacy
import json
import random
import torch
from spacy.training.example import Example
from tqdm import tqdm

# Enable GPU
if torch.cuda.is_available():
    spacy.require_gpu()
    print("✅ GPU enabled!")
else:
    print("⚠️ No GPU detected. Running on CPU.")

# Load base model
BASE_MODEL = "en_core_web_md"
nlp = spacy.load(BASE_MODEL)

# Add NER component if not present
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Load training data (Ensure you have a valid JSON file with 20k records)
TRAINING_DATA_FILE = "ner_training_data.json"
with open(TRAINING_DATA_FILE, "r", encoding="utf-8") as f:
    TRAIN_DATA = json.load(f)

# Add new entity labels
NER_LABELS = ["PHONE_MODEL", "MEMORY", "PRICE"]
for label in NER_LABELS:
    ner.add_label(label)

# Disable other pipeline components for faster training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

# Training configurations
N_ITER = 30  # Number of training iterations
BATCH_SIZE = 64  # Mini-batch size
BEST_MODEL_PATH = "best_trained_ner"  # Path to save best model
best_loss = float("inf")  # Track lowest loss

# Training loop
with nlp.disable_pipes(*other_pipes):  # Only train NER
    optimizer = nlp.resume_training()
    
    print("🚀 Starting training...")
    
    for epoch in tqdm(range(N_ITER), desc="Training Progress"):
        random.shuffle(TRAIN_DATA)  # Shuffle dataset
        
        losses = {}
        batch_loss = 0
        batch_examples = []
        
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            batch_examples.append(example)
            
            # Process in batches
            if len(batch_examples) >= BATCH_SIZE:
                nlp.update(batch_examples, drop=0.3, losses=losses)
                batch_loss += losses["ner"]
                batch_examples = []  # Reset batch
        
        # Train any remaining examples
        if batch_examples:
            nlp.update(batch_examples, drop=0.3, losses=losses)
            batch_loss += losses["ner"]
        
        print(f"Epoch {epoch+1}/{N_ITER}: Loss = {batch_loss:.4f}")
        
        # Save best model with lowest loss
        if batch_loss < best_loss:
            best_loss = batch_loss
            nlp.to_disk(BEST_MODEL_PATH)
            print(f"✅ New best model saved at {BEST_MODEL_PATH} (Loss: {best_loss:.4f})")

# Save final model
nlp.to_disk("final_trained_ner")
print("🎉 Training complete! Final model saved as 'final_trained_ner'")
