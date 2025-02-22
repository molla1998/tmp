import csv
import json
import itertools

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

# Generate dataset with all possible combinations
training_data = []
for template, phone_model, memory, price in itertools.product(templates, PHONE_MODELS, MEMORY_SIZES, PRICES):
    # Replace placeholders
    sentence = template.replace("{PHONE_MODEL}", phone_model)
    sentence = sentence.replace("{MEMORY}", memory)
    sentence = sentence.replace("{PRICE}", price)

    # Define entity positions
    entities = []
    for label, value in [("PHONE_MODEL", phone_model), ("MEMORY", memory), ("PRICE", price)]:
        start = sentence.find(value)
        if start != -1:
            entities.append((start, start + len(value), label))

    training_data.append((sentence, {"entities": entities}))

# Save dataset
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(training_data, f, indent=4, ensure_ascii=False)

print(f"✅ Training data saved to {OUTPUT_JSON} with {len(training_data)} examples.")
