from transformers import pipeline
import spacy
import re

# Load spaCy model for noun and dependency parsing
nlp_spacy = spacy.load("en_core_web_sm")

# Load Hugging Face NER pipeline
ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",  # State-of-the-art NER model
    tokenizer="dslim/bert-base-NER",
    device=0  # Use GPU (set to -1 for CPU)
)

def extract_keywords_in_order(query):
    # Preprocess the query
    query_lower = query.lower()

    # Extract noun phrases using spaCy
    doc = nlp_spacy(query)
    noun_phrases = [(chunk.text.lower(), chunk.start_char) for chunk in doc.noun_chunks]

    # Run the NER pipeline on the query
    entities = ner_pipeline(query)
    ner_keywords = [(ent["word"].lower(), ent["start"]) for ent in entities]

    # Extract numeric and alphanumeric patterns using regex
    numeric_patterns = [
        (match.group(), match.start()) for match in re.finditer(r"(\d+k|\d+\s?gb|\d+\s?mp|\d+\s?hz|₹?\d{1,3},?\d{1,3})", query_lower)
    ]

    # Extract adjectives (e.g., cheap, under, within) using regex
    adjectives = [
        (match.group(), match.start()) for match in re.finditer(r"\b(under|below|cheaper|cheap|within|atleast|max|min|expensive)\b", query_lower)
    ]

    # Combine all extractions and sort by their order in the query
    combined = noun_phrases + ner_keywords + numeric_patterns + adjectives
    sorted_keywords = sorted(combined, key=lambda x: x[1])  # Sort by position in text

    # Return only the words (remove position metadata)
    return [keyword[0] for keyword in sorted_keywords]

# Example Queries
queries = [
    "I want a cheap phone under ₹50k with 8GB RAM and 50MP camera",
    "Looking for a laptop with Core i7, SSD storage, and 16GB RAM within 70k",
    "Suggest me a TV under ₹30,000 with 120Hz refresh rate and 4K resolution",
    "Can you find me a tablet with 4GB RAM and under ₹15,000?",
]

# Run extraction on each query
for query in queries:
    print(f"Query: {query}")
    print("Extracted Keywords (Sequential Order):", extract_keywords_in_order(query))
    print("-" * 50)
