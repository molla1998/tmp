import pandas as pd
import requests
import ast


# ================================
# CONFIG
# ================================
API_URL = "YOUR_API_ENDPOINT"

def to_bool(x):
    return str(x).strip().lower() == "true"


# ---------- Normalize ----------
def normalize(text):
    return str(text).lower().strip()


# ---------- Overlap match ----------
def product_match(p1, p2):
    p1 = normalize(p1)
    p2 = normalize(p2)

    # substring match
    if p1 in p2 or p2 in p1:
        return True

    # token overlap
    t1 = set(p1.split())
    t2 = set(p2.split())

    return len(t1 & t2) > 0


# ---------- Call model ----------
def call_model(query):
    response = requests.post(API_URL, json={"query": query})
    return response.json()


# ---------- Extract predicted products ----------
def extract_products(pred_output):
    products = []

    for item in pred_output:
        # ✅ treat accessory also as product
        if item["label"] in ["product_name", "accessory"]:
            products.append({
                "product_name": item["text"],
                # ⚠️ Update this if your API provides is_main_product
                "is_main_product": False,
                "label": item["label"]
            })

    return products


# =========================================================
# MAIN EVALUATION
# =========================================================
def evaluate(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    rows = []

    TP = FP = FN = TN = 0
    FP_PRODUCT = FN_PRODUCT = 0

    for _, row in df.iterrows():
        query = row["query"]

        # ---- Parse GT ----
        gt_data = ast.literal_eval(row["gt"])

        gt_products = []
        for ent in gt_data["entities"]:
            gt_products.append({
                "product_name": ent["product_name"],
                "is_main_product": ent["is_main_product"]
            })

        # ---- Get prediction ----
        pred_output = call_model(query)
        pred_products = extract_products(pred_output)

        matched_pred_idx = set()

        # ---- Match GT → PRED ----
        for gt in gt_products:
            gt_name = gt["product_name"]
            gt_flag = to_bool(row["is_main_pdt_gt"])

            found_match = False

            for i, pred in enumerate(pred_products):
                if i in matched_pred_idx:
                    continue

                if product_match(gt_name, pred["product_name"]):
                    found_match = True
                    matched_pred_idx.add(i)

                    pred_flag = to_bool(row["is_main_pdt_pred"])
                    
                    # ---- Classification ----
                    if gt_flag and pred_flag:
                        if match:
                            status = "TP"
                            TP += 1
                        else:
                            status = "FN_PDT"
                    
                    elif not gt_flag and pred_flag:
                        status = "FP"
                        FP += 1
                    
                    elif gt_flag and not pred_flag:
                        status = "FN"
                        FN += 1
                    
                    else:
                        status = "TN"
                        TN += 1
                    rows.append({
                        "query": query,
                        "product_name_gt": gt_name,
                        "product_name_pred": pred["product_name"],
                        "is_main_product_gt": gt_flag,
                        "is_main_product_pred": pred_flag,
                        "status": status
                    })

                    break

            # ---- No matching product → FN_PRODUCT ----
            if not found_match:
                FN_PRODUCT += 1
                rows.append({
                    "query": query,
                    "product_name_gt": gt_name,
                    "product_name_pred": "",
                    "is_main_product_gt": gt_flag,
                    "is_main_product_pred": "",
                    "status": "FN_PRODUCT"
                })

        # ---- Extra predicted products → FP_PRODUCT ----
        for i, pred in enumerate(pred_products):
            if i not in matched_pred_idx:
                FP_PRODUCT += 1
                rows.append({
                    "query": query,
                    "product_name_gt": "",
                    "product_name_pred": pred["product_name"],
                    "is_main_product_gt": "",
                    "is_main_product_pred": pred["is_main_product"],
                    "status": "FP_PRODUCT"
                })

    # ---- Save analysis CSV ----
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)

    # ---- Metrics (ONLY classification, exclude *_PRODUCT) ----
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    print("\n===== FINAL METRICS (is_main_product only) =====")
    print(f"TP: {TP}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")
    print(f"TN: {TN}")
    print(f"FP_PRODUCT: {FP_PRODUCT}")
    print(f"FN_PRODUCT: {FN_PRODUCT}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")


# ---------- RUN ----------
if __name__ == "__main__":
    evaluate("input.csv", "analysis_output.csv")
