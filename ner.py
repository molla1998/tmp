from transformers import pipeline
import re

# Load Hugging Face NER pipeline with a pre-trained BERT model
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
    device=0  # Use GPU (set to -1 for CPU)
)

def extract_keywords(query):
    # Run the NER pipeline
    entities = ner_pipeline(query)

    # Initialize containers for extracted data
    keywords = []
    
    # Regex patterns for price and features
    price_patterns = [
        r"(under|within|below)\s?₹?\d+k?",  # e.g., "under 50k"
        r"₹?\d{1,3}[kK]?",                  # e.g., "50k", "₹50000"
    ]
    feature_patterns = [
        r"\d+gb\s?ram",                     # e.g., "8GB RAM"
        r"\d+\s?mp\s?camera",               # e.g., "50MP camera"
        r"\d+hz",                           # e.g., "120Hz"
        r"ssd|hdd",                         # e.g., "SSD", "HDD"
        r"core\s?i[3579]",                  # e.g., "Core i5"
    ]

    # Post-process NER entities
    temp_feature = ""
    for ent in entities:
        word = ent['word'].strip()

        # Check if the word matches product types or features
        if ent['entity'].startswith("B-MISC") and len(keywords) == 0:
            keywords.append(word)  # Add product type (e.g., phone, laptop)
        
        # Check for features (e.g., "8GB RAM", "50MP camera")
        elif any(re.search(pat, word.lower()) for pat in feature_patterns):
            if temp_feature:
                temp_feature += " " + word
            else:
                temp_feature = word
        elif any(re.search(pat, word.lower()) for pat in price_patterns):
            keywords.append(word)  # Add price-related keywords
        
        # If feature is complete, add it as one keyword
        if temp_feature and word.lower() not in temp_feature.lower():
            keywords.append(temp_feature)  # e.g., "8GB RAM", "50MP camera"
            temp_feature = ""

    # In case there's any leftover feature at the end of the query
    if temp_feature:
        keywords.append(temp_feature)

    # Return the list of keywords
    return list(set(keywords))  # Removing duplicates

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
