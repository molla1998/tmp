import csv
import time
from googletrans import Translator

# Initialize translator
translator = Translator()

# File paths
input_csv = "korean_texts.csv"  # Original CSV
output_csv = "translated_texts.csv"  # Temporary output CSV

# Open input file and create output file
with open(input_csv, mode="r", encoding="utf-8") as infile, \
     open(output_csv, mode="w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ["query_eng"]  # Add new column

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()  # Write new header

    for row in reader:
        korean_text = row.get("query_text", "").strip()

        if not korean_text:  # Skip empty rows
            row["query_eng"] = ""
        else:
            try:
                row["query_eng"] = translator.translate(korean_text, src='ko', dest='en').text
            except Exception as e:
                print(f"⚠ Error translating '{korean_text}': {e}")
                row["query_eng"] = "Translation Error"

            print(f"✅ Translated: {korean_text} → {row['query_eng']}")

        writer.writerow(row)  # Write row with new column
        time.sleep(1)  # Avoid hitting API rate limits

# Replace original file with updated file
import shutil
shutil.move(output_csv, input_csv)

print(f"\n✅ Translations added to '{input_csv}'")
