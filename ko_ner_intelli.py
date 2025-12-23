import csv
import requests
import json

INPUT_CSV = "input.csv"
OUTPUT_CSV = "output.csv"

API_URL = "https://your-api-url"   # <-- change this

HEADERS = {
    "Content-Type": "application/json"
    # Add auth headers if needed
    # "Authorization": "Bearer XXXX"
}

def call_api(query):
    payload = {
        "query": query
        # Add other body fields if your API needs them
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()

    data = response.json()

    ner = data.get("ner_output", "")

    # If ner_output is itself a JSON string → decode once
    if isinstance(ner, str):
        try:
            ner = json.loads(ner)
        except Exception:
            pass

    # Convert to single-line readable JSON string (Korean readable)
    return json.dumps(ner, ensure_ascii=False)

rows = []

with open(INPUT_CSV, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        query_value = row["query"]
        row["output"] = call_api(query_value)
        rows.append(row)

with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Done! Check output.csv")
