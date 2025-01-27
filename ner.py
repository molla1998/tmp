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
    product_type = None
    price = None
    features = []

    # Regex patterns for price and specifications
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
    for ent in entities:
        word = ent['word'].strip()
        if ent['entity'].startswith("B-MISC") and product_type is None:
            product_type = word  # Capture product type (e.g., phone, laptop)
        elif any(re.search(pat, word.lower()) for pat in price_patterns):
            price = word  # Capture price or related words
        elif any(re.search(pat, word.lower()) for pat in feature_patterns):
            features.append(word)  # Capture technical specifications

    # Post-processing for price and feature merging
    if price:
        # Normalize price (e.g., "50k" -> "50000")
        price = re.sub(r"k", "000", price.lower()).replace("₹", "")

    # Return structured data
    return {
        "Product Type": product_type or "Unknown",
        "Price": price or "Unknown",
        "Features": list(set(features)),  # Remove duplicates
    }

# Example Queries
queries = [
    "I want a cheap phone under ₹50k with 8GB RAM and 50MP camera",
    "Looking for a laptop with Core i7, SSD storage, and 16GB RAM within 70k",
    "Suggest me a TV under ₹30,000 with 120Hz refresh rate and 4K resolution",
]

# Run extraction on each query
for query in queries:
    print(f"Query: {query}")
    print("Extracted Information:", extract_keywords(query))
    print("-" * 50)
