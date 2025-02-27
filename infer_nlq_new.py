import pandas as pd
import spacy
from collections import Counter
import numpy as np

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define POS tags to exclude
EXCLUDE_POS = {"PUNCT", "SYM", "DET", "CCONJ", "PART", "SCONJ"}

# Function to calculate POS entropy
def pos_entropy(pos_counts, total_tokens):
    probabilities = [count / total_tokens for count in pos_counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

# Function to analyze text using both approaches
def analyze_text(text, noun_threshold=50, entropy_threshold1=1.5, entropy_threshold2=2.0):
    doc = nlp(text)
    
    # Filter out unwanted POS before counting total tokens
    filtered_tokens = [token for token in doc if token.pos_ not in EXCLUDE_POS]
    total_tokens = len(filtered_tokens)
    
    # Count POS only from meaningful tokens
    pos_counts = Counter([token.pos_ for token in filtered_tokens])
    
    noun_count = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
    noun_percentage = (noun_count / total_tokens) * 100 if total_tokens > 0 else 0
    entropy = pos_entropy(pos_counts, total_tokens) if total_tokens > 0 else 0
    
    # Approach 1
    if noun_percentage <= noun_threshold and entropy >= entropy_threshold1:
        classification1 = "🔹 Natural Language Query"
    else:
        classification1 = "🔹 Keyword-Heavy"

    # Approach 2
    if entropy >= entropy_threshold2:
        classification2 = "🔹 Natural Language Query"
    else:
        classification2 = "🔹 Keyword-Heavy"
    
    return classification1, classification2

# Function to process CSV file
def process_csv(input_csv, output_csv):
    # Load CSV
    df = pd.read_csv(input_csv)

    # Ensure "query" column exists
    if "query" not in df.columns:
        raise ValueError("The input CSV must contain a 'query' column.")

    # Apply inference
    df[["Approach 1", "Approach 2"]] = df["query"].apply(lambda x: pd.Series(analyze_text(str(x))))

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    input_path = input("Enter the path to the input CSV file: ").strip()
    output_path = "results.csv"  # Change if needed

    try:
        process_csv(input_path, output_path)
    except Exception as e:
        print(f"Error: {e}")
