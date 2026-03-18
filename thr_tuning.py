import pandas as pd
import json
from difflib import SequenceMatcher
from collections import defaultdict
import numpy as np

############################################
# CONFIG
############################################

THRESHOLDS = np.arange(0.2, 0.8, 0.05)
PRODUCT_MATCH_THRESHOLD = 0.75  # fuzzy match threshold

############################################
# SOFT PRODUCT MATCH (NO EMBEDDINGS)
############################################

def fuzzy_match(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def is_same_product(pred, gt):
    return fuzzy_match(pred, gt) >= PRODUCT_MATCH_THRESHOLD

############################################
# SIMPLE TOKEN OVERLAP (OPTIONAL BOOST)
############################################

def token_overlap(a, b):
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())

    if not set_a or not set_b:
        return 0

    return len(set_a & set_b) / min(len(set_a), len(set_b))

def product_match_score(pred, gt):
    return max(
        fuzzy_match(pred, gt),
        token_overlap(pred, gt)
    )

############################################
# NORMALIZE VALUES
############################################

def normalize(val):
    return val.lower().strip()

############################################
# EXTRACT PRODUCTS FROM NER
############################################

def get_products(entities):
    return [e for e in entities if e["label"] == "product_name"]

############################################
# FAKE LINKER (since you already have scores)
# Replace this with your real linker output
############################################

def generate_pairs(entities):
    """
    Returns:
    [(attribute, product, score)]
    """
    products = get_products(entities)
    attributes = [e for e in entities if e["label"] != "product_name"]

    pairs = []

    for attr in attributes:
        for prod in products:
            # ⚠️ Replace this with your real score
            score = np.random.uniform(0.2, 0.8)

            pairs.append({
                "attribute": attr["label"],
                "value": attr["text"],
                "product": prod["text"],
                "score": score
            })

    return pairs

############################################
# EVALUATION STORAGE
############################################

attribute_scores = defaultdict(list)

############################################
# MAIN LOOP
############################################

def process_csv(path):

    df = pd.read_csv(path)

    for _, row in df.iterrows():

        ner_entities = json.loads(row["ner_entities"])
        gt = json.loads(row["gt_entities"])

        pairs = generate_pairs(ner_entities)

        gt_product = gt.get("product_name", "").lower()

        for pair in pairs:

            attr = pair["attribute"]
            pred_val = normalize(pair["value"])
            pred_prod = pair["product"]
            score = pair["score"]

            gt_val = gt.get(attr, None)

            # skip if GT doesn't have this attribute
            if gt_val is None:
                label = 0
            else:
                gt_val = normalize(gt_val)

                # product match
                match_score = product_match_score(pred_prod, gt_product)

                if match_score >= PRODUCT_MATCH_THRESHOLD:
                    if pred_val == gt_val:
                        label = 1  # TRUE POSITIVE
                    else:
                        label = 0  # WRONG VALUE
                else:
                    label = 0  # WRONG PRODUCT

            attribute_scores[attr].append((score, label))

############################################
# METRICS
############################################

def compute_metrics(pairs, threshold):

    tp = fp = fn = 0

    for score, label in pairs:

        pred = score >= threshold

        if pred and label == 1:
            tp += 1
        elif pred and label == 0:
            fp += 1
        elif not pred and label == 1:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

############################################
# FIND BEST THRESHOLD PER ATTRIBUTE
############################################

def find_best_thresholds():

    results = {}

    for attr, pairs in attribute_scores.items():

        best_f1 = 0
        best_t = 0

        for t in THRESHOLDS:

            p, r, f1 = compute_metrics(pairs, t)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        results[attr] = {
            "threshold": round(best_t, 3),
            "f1": round(best_f1, 3)
        }

    return results

############################################
# RUN
############################################

if __name__ == "__main__":

    process_csv("data.csv")

    best_thresholds = find_best_thresholds()

    print("\nBest Thresholds Per Attribute:\n")

    for attr, res in best_thresholds.items():
        print(f"{attr} → threshold={res['threshold']} | f1={res['f1']}")
