import csv
from googletrans import Translator

# Function to split a list into smaller batches
def chunk_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

# Function to translate Korean to English in batches
def batch_translate_korean_to_english(texts, batch_size=10):
    translator = Translator()
    all_translations = []
    
    for batch in chunk_list(texts, batch_size):
        translations = translator.translate(batch, src='ko', dest='en')
        all_translations.extend([t.text for t in translations])
    
    return all_translations

# Function to read Korean text from a CSV file
def read_korean_from_csv(input_csv):
    korean_texts = []
    with open(input_csv, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip header if present
        for row in reader:
            if row:  # Avoid empty rows
                korean_texts.append(row[0])  # Assuming Korean text is in the first column
    return korean_texts

# Function to write translations to a new CSV file
def write_translations_to_csv(output_csv, korean_texts, english_texts):
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Korean", "English"])  # Write header
        for k, e in zip(korean_texts, english_texts):
            writer.writerow([k, e])
    print(f"Translations saved to {output_csv}")

# File paths
input_csv = "korean_texts.csv"  # Your input CSV file with Korean text
output_csv = "translated_texts.csv"  # Output file with translations

# Load Korean text from CSV
korean_sentences = read_korean_from_csv(input_csv)

# Translate and save to CSV (with batching)
if korean_sentences:
    translated_sentences = batch_translate_korean_to_english(korean_sentences, batch_size=10)
    write_translations_to_csv(output_csv, korean_sentences, translated_sentences)
else:
    print("No text found in the CSV file.")
