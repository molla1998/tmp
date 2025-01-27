from transformers import pipeline
import re

# Load the Hugging Face NER pipeline with a pretrained model
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
    device=0  # Use GPU; set to -1 for CPU
)

def extract_keywords(query):
    # Run NER pipeline
    entities = ner_pipeline(query)

    # Initialize lists for keyword extraction
    keywords = []
    temp_phrase = ""

    # Process NER entities
    for ent in entities:
        word = ent["word"]
        entity_type = ent["entity"]

        # Product types, adjectives, and features based on NER
        if entity_type in ["B-MISC", "I-MISC"]:  # Miscellaneous entities (e.g., product types, features)
            temp_phrase += word if temp_phrase == "" else f" {word}"
        elif entity_type in ["B-ORG", "B-PER", "B-LOC"]:  # Organizational names, locations, or personal names
            keywords.append(word)

        # End of a phrase
        if temp_phrase and not word.startswith("##"):  # Finalize and reset
            keywords.append(temp_phrase.strip())
            temp_phrase = ""

    # Extract numeric or price-related terms from the query
    price_matches = re.findall(r"(\d+k|\d+\s?gb|\d+\s?mp|\d+\s?hz|₹?\d{1,3},?\d{1,3})", query.lower())
    price_matches = [match.replace(" ", "") for match in price_matches]  # Clean up spaces in terms like "8 GB"

    # Extract adjectives (e.g., cheap, under, within)
    adjective_matches = re.findall(r"\b(under|below|cheaper|cheap|within|atleast|max|min|expensive)\b", query.lower())

    # Combine all keywords
    all_keywords = list(set(keywords + price_matches + adjective_matches))  # Remove duplicates
    return all_keywords

# Example Queries
queries = [
    "I want a cheap phone under ₹50k with 8GB RAM and 50MP camera",
    "Looking for a laptop with Core i7, SSD storage, and 16GB RAM within 70k",
    "Suggest me a TV under ₹30,000 with 120Hz refresh rate and 4K resolution",
]

# Run extraction on each query
for query in queries:
    print(f"Query: {query}")
    print("Extracted Keywords:", extract_keywords(query))
    print("-" * 50)
