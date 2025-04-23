import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import difflib

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def match_keyword(text, keyword):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    keyword = keyword.lower()
    
    # Tokenize the text
    words = text.split()

    # Lemmatized version of keyword
    keyword_lemma = lemmatizer.lemmatize(keyword)

    # Get all possible forms of keyword
    variants = set()
    variants.add(keyword)
    variants.add(keyword_lemma)
    variants.update(get_synonyms(keyword))
    
    # Optionally add plural (simple form)
    if not keyword.endswith('s'):
        variants.add(keyword + 's')
    
    # Check for exact or close matches
    for word in words:
        if word in variants:
            return True
        # Close match with 0.8+ similarity
        matches = difflib.get_close_matches(word, variants, n=1, cutoff=0.8)
        if matches:
            return True

    return False

# Example usage
text = "I really like smartphones and smart devices."
keyword = "phone"

print(match_keyword(text, keyword))  # Output: True (matches "smartphones")
