import pandas as pd
import json

# fields you want to extract
TARGET_FIELDS = ["memory", "product_name"]

def transform_entities(json_str):
    try:
        data = json.loads(json_str)
        result = []

        for item in data.get("entities", []):
            for field in TARGET_FIELDS:
                if field in item:
                    text = item[field]
                    result.append({
                        "label": field,
                        "text": text,
                        "start": 0,
                        "end": len(text)
                    })

        return json.dumps(result)

    except Exception as e:
        return "[]"  # fallback if parsing fails


# load CSV
df = pd.read_csv("input.csv")

# apply transformation
df["output"] = df["input"].apply(transform_entities)

# save to new CSV
df.to_csv("output.csv", index=False)

print("Done! Check output.csv")
