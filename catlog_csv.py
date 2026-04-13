import pandas as pd
from rank_bm25 import BM25Okapi
import re

# =========================
# LOAD PRODUCT CSV
# =========================
df_products = pd.read_csv("products.csv")  
# expected columns: id, product_description

# =========================
# PREPROCESS
# =========================
def preprocess(text):
    if pd.isna(text):
        return []
    return str(text).lower().split()

corpus = df_products["product_description"].apply(preprocess).tolist()
bm25 = BM25Okapi(corpus)

# =========================
# NORMALIZATION
# =========================
def normalize(text):
    return str(text).lower().strip()

# =========================
# DUMMY NER (REPLACE WITH YOUR MODEL)
# =========================
def ner_model(query):
    query = query.lower()

    entity = {
        "id": 1,
        "product_name": "phone",
        "is_main_product": True
    }

    if "5g" in query:
        entity["connections"] = "5g"

    match = re.search(r"\d+gb", query)
    if match:
        entity["memory"] = match.group()

    return {
        "intent": "product_search",
        "entities": [entity]
    }

# =========================
# BUILD NER QUERY
# =========================
def build_ner_query(entity):
    parts = []

    # Boost product name
    if entity.get("product_name"):
        parts.extend([entity["product_name"]] * 3)

    # Add attributes
    for k, v in entity.items():
        if k not in ["id", "is_main_product", "product_name"] and v:
            parts.append(v)

    return " ".join(parts)

# =========================
# HYBRID QUERY
# =========================
def build_hybrid_query(raw_query, ner_output):
    entities = ner_output.get("entities", [])

    main = next((e for e in entities if str(e.get("is_main_product")).lower() == "true"), None)
    if not main and entities:
        main = entities[0]

    ner_query = build_ner_query(main) if main else ""

    return raw_query + " " + ner_query

# =========================
# SEARCH FUNCTION
# =========================
def search(query, top_n=5):
    tokens = preprocess(query)
    scores = bm25.get_scores(tokens)

    ranked = sorted(
        zip(df_products["product_name"], scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_n]

# =========================
# RUN PIPELINE
# =========================
def run_pipeline(input_csv, output_csv):
    df_queries = pd.read_csv(input_csv)  # column: query

    results = []

    for _, row in df_queries.iterrows():
        raw_query = row["query"]

        ner_output = ner_model(raw_query)
        final_query = build_hybrid_query(raw_query, ner_output)

        top_results = search(final_query, top_n=5)

        results.append({
    "query": raw_query,
    "final_query": final_query,
    "top5_products": ", ".join([name for name, _ in top_results]),
    "top5_scores": ", ".join([f"{score:.4f}" for _, score in top_results])
})

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)

    print(f"✅ Results saved to {output_csv}")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run_pipeline("queries.csv", "results.csv")
