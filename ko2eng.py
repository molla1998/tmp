import csv
from googletrans import Translator

# Function to read Korean text from a CSV file
def read_korean_from_csv(input_csv):
    korean_texts = []
    with open(input_csv, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if present
        for row in reader:
            if row:  # Avoid empty rows
                korean_texts.append(row[0])  # Assuming Korean text is in the first column
    return korean_texts

# Function to translate Korean to English
def batch_translate_korean_to_english(texts):
    translator = Translator()
    translations = translator.translate(texts, src='ko', dest='en')
    return [t.text for t in translations]  # Extract English text

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

# Translate and save to CSV
if korean_sentences:
    translated_sentences = batch_translate_korean_to_english(korean_sentences)
    write_translations_to_csv(output_csv, korean_sentences, translated_sentences)
else:
    print("No text found in the CSV file.")
