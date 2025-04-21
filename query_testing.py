import csv
import subprocess
import pandas as pd
import json
import re

# Config
input_csv_path = "input.csv"
output_csv_path = "output.csv"
command_template = "your_cli_tool --query \"{query}\""

def parse_output(output):
    try:
        # Convert to proper JSON format
        json_like = re.sub(r'(\w+):', r'"\1":', output.strip())  # adds quotes to keys
        json_like = json_like.replace("true", "True").replace("false", "False")
        data = eval(json_like)  # or use json.loads(json_like) if it's fully valid JSON

        ans = bool(data.get("ans", False))
        query = data.get("query", "")
        inf_time_raw = data.get("inf_time", "0ms").replace("ms", "")
        inf_time = float(inf_time_raw)

        return {
            "ans": ans,
            "inference_time": inf_time,
            "query_text": query
        }

    except Exception as e:
        return {
            "ans": False,
            "inference_time": -1,
            "query_text": "",
            "error": str(e)
        }

def run_command(query):
    cmd = command_template.format(query=query)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return parse_output(result.stdout)
    except Exception as e:
        return {
            "ans": False,
            "inference_time": -1,
            "query_text": query,
            "error": str(e)
        }

# Load input CSV
df = pd.read_csv(input_csv_path)

# Run CLI commands
results = []
for query in df['query_text']:
    res = run_command(query)
    results.append(res)

# Save to output CSV
output_df = pd.DataFrame(results)
output_df.to_csv(output_csv_path, index=False)

print(f"Saved results to {output_csv_path}")
