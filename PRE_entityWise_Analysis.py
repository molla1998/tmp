import pandas as pd
import requests
from rapidfuzz import fuzz
import time

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "https://your-api-endpoint.com/predict"
HEADERS = {"Content-Type": "application/json"}
PRODUCT_MATCH_THRESHOLD = 80
SLEEP_TIME = 0.1  # avoid rate limits


# -----------------------------
# SOFT MATCH FUNCTION
# -----------------------------
def product_match(gt, pred, threshold=PRODUCT_MATCH_THRESHOLD):
    gt = str(gt).lower().strip()
    pred = str(pred).lower().strip()

    if gt == pred:
        return True
    if gt in pred or pred in gt:
        return True
    if fuzz.ratio(gt, pred) >= threshold:
        return True
    return False


# -----------------------------
# API CALL
# -----------------------------
def get_prediction(query):
    payload = {"query": query}

    try:
        response = requests.post(API_URL, json=payload, headers=HEADERS)
        response.raise_for_status()
        data = response.json()

        # expected format:
        # {
        #   "intent": "...",
        #   "entities": [
        #       {"product_name": "...", "color": "..."}
        #   ]
        # }

        return data.get("entities", [])

    except Exception as e:
        print(f"API error for query: {query} -> {e}")
        return []


# -----------------------------
# LOAD GT
# -----------------------------
gt_df = pd.read_csv("gt.csv")

# ensure clean
gt_df["query"] = gt_df["query"].astype(str)


# -----------------------------
# EVALUATION
# -----------------------------
results = []

for query in gt_df["query"].unique():

    gt_rows = gt_df[gt_df["query"] == query].reset_index(drop=True)

    # 🔥 CALL API HERE
    pred_entities = get_prediction(query)

    pred_rows = pd.DataFrame(pred_entities)

    if pred_rows.empty:
        pred_rows = pd.DataFrame(columns=["product_name", "color"])

    matched_gt = set()
    matched_pred = set()

    # -------- MATCHING --------
    for i, gt_row in gt_rows.iterrows():
        best_j = None

        for j, pred_row in pred_rows.iterrows():
            if j in matched_pred:
                continue

            if product_match(gt_row["product_name"], pred_row.get("product_name")):
                best_j = j
                break

        if best_j is not None:
            matched_gt.add(i)
            matched_pred.add(best_j)

            pred_row = pred_rows.loc[best_j]

            # PRODUCT STATUS
            product_status = "TP"

            # COLOR STATUS
            if str(gt_row["value"]).lower() == str(pred_row.get("color")).lower():
                color_status = "TP"
            else:
                color_status = "FP"

            results.append({
                "query": query,
                "product_gt": gt_row["product_name"],
                "product_pred": pred_row.get("product_name"),
                "product_status": product_status,
                "color_gt": gt_row["value"],
                "color_pred": pred_row.get("color"),
                "color_status": color_status
            })

        else:
            # FN
            results.append({
                "query": query,
                "product_gt": gt_row["product_name"],
                "product_pred": None,
                "product_status": "FN",
                "color_gt": gt_row["value"],
                "color_pred": None,
                "color_status": "FN"
            })

    # -------- EXTRA PREDICTIONS (FP) --------
    for j, pred_row in pred_rows.iterrows():
        if j not in matched_pred:
            results.append({
                "query": query,
                "product_gt": None,
                "product_pred": pred_row.get("product_name"),
                "product_status": "FP",
                "color_gt": None,
                "color_pred": pred_row.get("color"),
                "color_status": "FP"
            })

    time.sleep(SLEEP_TIME)


# -----------------------------
# RESULT DF
# -----------------------------
result_df = pd.DataFrame(results)

# Save detailed results
result_df.to_csv("evaluation_output.csv", index=False)

print("\nSaved row-level results to evaluation_output.csv")


# -----------------------------
# METRICS
# -----------------------------
def compute_metrics(df, col):
    tp = (df[col] == "TP").sum()
    fp = (df[col] == "FP").sum()
    fn = (df[col] == "FN").sum()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return tp, fp, fn, precision, recall, f1


# PRODUCT METRICS
p_tp, p_fp, p_fn, p_prec, p_rec, p_f1 = compute_metrics(result_df, "product_status")

# COLOR METRICS
c_tp, c_fp, c_fn, c_prec, c_rec, c_f1 = compute_metrics(result_df, "color_status")


# -----------------------------
# PRINT METRICS
# -----------------------------
print("\n=== PRODUCT METRICS ===")
print(f"TP={p_tp}, FP={p_fp}, FN={p_fn}")
print(f"Precision={p_prec:.4f}, Recall={p_rec:.4f}, F1={p_f1:.4f}")

print("\n=== COLOR METRICS ===")
print(f"TP={c_tp}, FP={c_fp}, FN={c_fn}")
print(f"Precision={c_prec:.4f}, Recall={c_rec:.4f}, F1={c_f1:.4f}")
