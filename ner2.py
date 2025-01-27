import spacy
import re

# Load spaCy model for noun and dependency parsing
nlp_spacy = spacy.load("en_core_web_sm")  # You can use "en_core_web_trf" for transformers

# List of common pronouns to filter out
PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", 
            "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs"}

def extract_keywords_in_order(query):
    # Preprocess the query
    query_lower = query.lower()

    # Extract noun phrases using spaCy (focus on nouns)
    doc = nlp_spacy(query)
    
    # Extract noun chunks (e.g., "phone", "tv", "laptop") and individual nouns
    noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]

    # Also consider individual nouns (since "tv" and "phone" may not be in noun chunks)
    nouns = [token.text.lower() for token in doc if token.pos_ == "NOUN"]

    # Combine noun phrases and individual nouns and remove duplicates
    combined_nouns = list(set(noun_phrases + nouns))

    # Run the NER pipeline on the query (optional, based on requirement)
    # entities = ner_pipeline(query)  # Uncomment if you want to include entities like "iphone"
    # ner_keywords = [ent["word"].lower() for ent in entities]

    # Extract numeric and alphanumeric patterns using regex (e.g., "50k", "8GB", "50MP")
    numeric_patterns = [
        match.group() for match in re.finditer(r"(\d+k|\d+\s?gb|\d+\s?mp|\d+\s?hz|₹?\d{1,3},?\d{1,3})", query_lower)
    ]

    
    # Combine all extractions and filter out pronouns
    filtered_keywords = list(set(combined_nouns + numeric_patterns))  # Remove duplicates

    # Return only relevant keywords
    return [keyword for keyword in filtered_keywords if keyword not in PRONOUNS]

# Example Queries
queries = [
    "I want a cheap phone under ₹50k with 8GB RAM and 50MP camera",
    "Looking for a laptop with Core i7, SSD storage, and 16GB RAM within 70k",
    "Suggest me a TV under ₹30,000 with 120Hz refresh rate and 4K resolution",
    "Can you find me a tablet with 4GB RAM and under ₹15,000?"
]

# Run extraction on each query
for query in queries:
    print(f"Query: {query}")
    print("Extracted Keywords (Nouns and Features):", extract_keywords_in_order(query))
    print("-" * 50)
