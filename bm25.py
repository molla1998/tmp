import json
import pandas as pd
from rank_bm25 import BM25Okapi

# =========================
# LOAD CATALOG
# =========================
with open("catalog.json") as f:
    data = json.load(f)

products = list(data.values()) if isinstance(data, dict) else data

# =========================
# NORMALIZATION HELPERS
# =========================
def normalize(text):
    return str(text).lower().replace(" ", "").strip()

def is_true(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() == "true"
    if isinstance(val, int):
        return val == 1
    return False

# =========================
# PREPROCESS FOR BM25
# =========================
def preprocess(product):
    text_parts = []

    text_parts.append(product.get("productDisplayname", ""))
    text_parts.append(product.get("category", ""))
    text_parts.append(product.get("sub_category", ""))
    text_parts.append(product.get("sub_genre", ""))
    text_parts.append(product.get("product_description", ""))

    # list fields
    text_parts.extend(product.get("search_keywords", []))
    text_parts.extend(product.get("all_attributes_value", []))

    return " ".join(text_parts).lower().split()

corpus = [preprocess(p) for p in products]
bm25 = BM25Okapi(corpus)

# =========================
# PIPELINE 1: RAW BM25
# =========================
def search_raw(query, top_n=5):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    ranked = sorted(zip(products, scores), key=lambda x: x[1], reverse=True)

    return [
        {"name": p["productDisplayname"], "score": s}
        for p, s in ranked[:top_n]
    ]

# =========================
# DUMMY NER (REPLACE)
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

    if "gb" in query:
        import re
        match = re.search(r"\d+gb", query)
        if match:
            entity["memory"] = match.group()

    return {
        "intent": "product_search",
        "entities": [entity]
    }

# =========================
# BUILD QUERY FROM NER
# =========================
def build_query(entity):
    parts = []

    if entity.get("product_name"):
        parts.extend([entity["product_name"]] * 3)

    for k, v in entity.items():
        if k not in ["id", "is_main_product", "product_name"] and v:
            parts.append(v)

    return " ".join(parts)

# =========================
# GENERIC CATEGORY FILTER
# =========================
def filter_category(products, entity):
    name = entity.get("product_name", "")
    tokens = name.lower().split()

    filtered = []

    for p in products:
        category_text = " ".join([
            p.get("category", ""),
            p.get("sub_category", ""),
            p.get("sub_genre", "")
        ]).lower()

        if any(token in category_text for token in tokens):
            filtered.append(p)

    return filtered if filtered else products

# =========================
# ATTRIBUTE BOOST
# =========================
def boost_attributes(results, entity):
    boosted = []

    for r in results:
        product = r["product"]
        score = r["score"]

        attrs = [normalize(a) for a in product.get("all_attributes_value", [])]

        match_count = 0

        for k, v in entity.items():
            if k not in ["id", "is_main_product", "product_name"] and v:
                if normalize(v) in attrs:
                    match_count += 1

        if match_count > 0:
            score *= (1 + 0.3 * match_count)

        boosted.append({"product": product, "score": score})

    return boosted

# =========================
# KEYWORD BOOST
# =========================
def boost_keywords(results, entity):
    boosted = []

    for r in results:
        product = r["product"]
        score = r["score"]

        keywords = [normalize(k) for k in product.get("search_keywords", [])]

        for k, v in entity.items():
            if v and normalize(v) in keywords:
                score *= 1.2

        boosted.append({"product": product, "score": score})

    return boosted

# =========================
# SECONDARY ENTITY BOOST
# =========================
def boost_secondary(results, entities):
    secondary = [
        e.get("product_name", "").lower()
        for e in entities
        if not is_true(e.get("is_main_product"))
    ]

    boosted = []

    for r in results:
        product = r["product"]
        score = r["score"]

        text = (
            product.get("productDisplayname", "") + " " +
            product.get("product_description", "")
        ).lower()

        for term in secondary:
            if term and term in text:
                score *= 1.2

        boosted.append({"product": product, "score": score})

    return boosted

# =========================
# PIPELINE 2: NER → SMART SEARCH
# =========================
def search_ner(query, top_n=5):
    ner_output = ner_model(query)
    entities = ner_output.get("entities", [])

    # normalize boolean
    for e in entities:
        e["is_main_product"] = is_true(e.get("is_main_product"))

    main = next((e for e in entities if e["is_main_product"]), None)

    if not main:
        main = entities[0] if entities else {}

    query_built = build_query(main)

    filtered_products = filter_category(products, main)

    tokens = query_built.lower().split()
    scores = bm25.get_scores(tokens)

    results = [
        {"product": p, "score": s}
        for p, s in zip(products, scores)
        if p in filtered_products
    ]

    results = boost_attributes(results, main)
    results = boost_keywords(results, main)
    results = boost_secondary(results, entities)

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return [
        {"name": r["product"]["productDisplayname"], "score": r["score"]}
        for r in results[:top_n]
    ]

# =========================
# BATCH RUN + CSV OUTPUT
# =========================
def run_batch(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    rows = []

    for _, row in df.iterrows():
        query = row["query"]

        raw = search_raw(query)
        ner = search_ner(query)

        rows.append({
            "query": query,
            "raw_top5": ", ".join([r["name"] for r in raw]),
            "ner_top5": ", ".join([r["name"] for r in ner]),
            "raw_avg_score": sum(r["score"] for r in raw) / len(raw),
            "ner_avg_score": sum(r["score"] for r in ner) / len(ner)
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)

    print(f"✅ Saved to {output_csv}")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run_batch("queries.csv", "results.csv")
