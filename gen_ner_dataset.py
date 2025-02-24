import csv
import json
import random

# Load sentence templates
TEMPLATE_FILE = "/mnt/data/sentence_templates.csv"
OUTPUT_JSON = "ner_training_data.json"

# Lists of PHONE_MODEL, MEMORY, and PRICE
PHONE_MODELS = ["iPhone 14", "Galaxy S23", "Pixel 7", "OnePlus 11"]
MEMORY_SIZES = ["8GB RAM", "16GB RAM", "12GB RAM", "6GB"]
PRICES = ["10,000", "$100k", "₹50k", "75,000.00"]

# Load templates
templates = []
with open(TEMPLATE_FILE, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    templates = [row[0] for row in reader]

# Ensure all values appear at least once
base_combinations = [
    (random.choice(templates), model, memory, price)
    for model in PHONE_MODELS
    for memory in MEMORY_SIZES
    for price in PRICES
]

# Generate additional records up to 200k
additional_samples = [
    (random.choice(templates), random.choice(PHONE_MODELS), random.choice(MEMORY_SIZES), random.choice(PRICES))
    for _ in range(200000 - len(base_combinations))
]

# Final dataset
samples = base_combinations + additional_samples
random.shuffle(samples)  # Shuffle for randomness

# Create training data
training_data = []
for template, phone_model, memory, price in samples:
    sentence = template.replace("{PHONE_MODEL}", phone_model).replace("{MEMORY}", memory).replace("{PRICE}", price)

    # Find entity positions
    entities = []
    for label, value in [("PHONE_MODEL", phone_model), ("MEMORY", memory), ("PRICE", price)]:
        start = sentence.find(value)
        end = start + len(value)
        if start != -1 and all(not (s < end and start < e) for s, e, _ in entities):  # Avoid overlap
            entities.append((start, end, label))

    training_data.append((sentence, {"entities": entities}))

# Save dataset
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(training_data, f, indent=4, ensure_ascii=False)

print(f"✅ Training data saved to {OUTPUT_JSON} with {len(training_data)} examples.")
