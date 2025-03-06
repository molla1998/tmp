import pandas as pd
import torch
from gliner import GLiNER

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model on GPU if available
model = GLiNER.from_pretrained("numind/NuNER_Zero").to(device)

# Load CSV files
queries_df = pd.read_csv("queries.csv")  # Column name assumed: "query"
labels_df = pd.read_csv("labels.csv")    # Column name assumed: "label"

queries = queries_df["query"].tolist()
labels = labels_df["label"].tolist()

# Function to merge entities with start/end indices
def merge_entities(entities, text):
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity["label"] == current["label"] and (
            next_entity["start"] == current["end"] + 1 or next_entity["start"] == current["end"]
        ):
            current["text"] = text[current["start"]: next_entity["end"]].strip()
            current["end"] = next_entity["end"]
        else:
            merged.append(current)
            current = next_entity
    merged.append(current)
    return merged

# Store results
results = []

# Process labels one at a time
for label in labels:
    entity_count = 0  # Count how many queries contain this label

    # Process queries one by one (NO BATCHING)
    for query in queries:
        entities = model.predict_entities(query, [label.lower()])
        merged_entities = merge_entities(entities, query)

        if merged_entities:
            entity_count += 1  # Count queries where entity is found

        for entity in merged_entities:
            results.append([query, label, entity["text"], entity["start"], entity["end"]])

    # Print entity count per label
    print(f"Entity '{label}' found in {entity_count} queries.")

# Save results to CSV
output_df = pd.DataFrame(results, columns=["Query", "Label", "Entity", "Start", "End"])
output_df.to_csv("entity_results.csv", index=False)

print("Processing completed. Results saved to 'entity_results.csv'.")
