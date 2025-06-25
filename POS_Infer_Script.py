import json
from collections import Counter

# Dummy function to simulate extraction logic (replace with your actual logic)
def extract_entities(sentence):
    # This should be replaced with your real extraction logic using spaCy
    return {
        "Primary main noun": "dummy",
        "Primary prev nouns": "",
        "Adj ref Primary noun": "",
        "ADP ref Primary noun": "",
        "Verb ref Primary noun": "",
        "Secondary main noun": "dummy",
        "Secondary next nouns": "",
        "Adj ref Secondary noun": "",
        "ADP ref Secondary noun": "",
        "Verb ref Secondary noun": ""
    }

# Load generated dataset
with open("generated_ground_truth_dataset.json") as f:
    dataset = json.load(f)

# Counters for category-wise matches
match_counters = Counter()
total_counters = Counter()

# Categories to check
categories = [
    "Primary main noun",
    "Primary prev nouns",
    "Adj ref Primary noun",
    "ADP ref Primary noun",
    "Verb ref Primary noun",
    "Secondary main noun",
    "Secondary next nouns",
    "Adj ref Secondary noun",
    "ADP ref Secondary noun",
    "Verb ref Secondary noun"
]

# Evaluate
for item in dataset:
    sentence = item["text"]
    expected = item["expected"]
    extracted = extract_entities(sentence)

    for cat in categories:
        total_counters[cat] += 1
        if expected[cat] == extracted[cat]:
            match_counters[cat] += 1

# Show results
print("Category-wise perfect match counts:")
for cat in categories:
    print(f"{cat}: {match_counters[cat]} / {total_counters[cat]}")
