import pandas as pd
import torch
from gliner import GLiNER

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model on GPU if available
model = GLiNER.from_pretrained("numind/NuNER_Zero").to(device)

# Load queries and labels from CSV
queries_df = pd.read_csv("queries.csv")  # Column name assumed: "query"
labels_df = pd.read_csv("labels.csv")    # Column name assumed: "label"

# Extract lists
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
            current["text"] = text[current["start"] : next_entity["end"]].strip()
            current["end"] = next_entity["end"]
        else:
            merged.append(current)
            current = next_entity
    merged.append(current)
    return merged

# Prepare results storage
results = []

# Process labels one at a time
for label in labels:
    entity_count = 0
    batch_size = 32  # Adjust based on GPU memory
    num_batches = (len(queries) + batch_size - 1) // batch_size  # Ensure all queries are covered

    # Process in batches
    for i in range(num_batches):
        batch_queries = queries[i * batch_size : (i + 1) * batch_size]
        
        # Run entity recognition on batch
        batch_entities = model.predict_entities(batch_queries, [label.lower()])

        # Collect results
        for query, entities in zip(batch_queries, batch_entities):
            merged_entities = merge_entities(entities, query)

            for entity in merged_entities:
                results.append([query, label, entity["text"], entity["start"], entity["end"]])
                entity_count += 1

    # Print entity count per label
    print(f"Entity '{label}' found in {entity_count} queries.")

# Save results to CSV
output_df = pd.DataFrame(results, columns=["Query", "Label", "Entity", "Start", "End"])
output_df.to_csv("entity_results.csv", index=False)

print("Processing completed. Results saved to 'entity_results.csv'.")
