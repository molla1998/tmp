import pandas as pd


# ---------- Normalize text ----------
def normalize(text):
    return str(text).lower().strip()


# ---------- Overlap check ----------
def product_match(p1, p2):
    p1 = normalize(p1)
    p2 = normalize(p2)

    # substring match
    if p1 in p2 or p2 in p1:
        return True

    # token overlap
    tokens1 = set(p1.split())
    tokens2 = set(p2.split())

    return len(tokens1 & tokens2) > 0


# =========================================================
# 1️⃣ STRICT (ROW-LEVEL with overlap)
# =========================================================
def strict_metrics(df):
    TP = FP = FN = TN = 0

    for _, row in df.iterrows():
        gt_flag = str(row["is_main_pdt_gt"]).lower() == "true"
        pred_flag = str(row["is_main_pdt_pred"]).lower() == "true"

        gt_product = row["product_gt"]
        pred_product = row["product_pred"]

        match = product_match(gt_product, pred_product)

        if gt_flag and pred_flag and match:
            TP += 1
        elif pred_flag and (not gt_flag or not match):
            FP += 1
        elif gt_flag and (not pred_flag or not match):
            FN += 1
        else:
            TN += 1

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# =========================================================
# 2️⃣ AT-LEAST-ONE CORRECT (QUERY-LEVEL with overlap)
# =========================================================
def atleast_one_metrics(df):
    TP = FP = FN = 0

    grouped = df.groupby("query")

    for query, group in grouped:
        gt_any = any(str(x).lower() == "true" for x in group["is_main_pdt_gt"])
        pred_any = any(str(x).lower() == "true" for x in group["is_main_pdt_pred"])

        correct_any = False

        for _, row in group.iterrows():
            gt_flag = str(row["is_main_pdt_gt"]).lower() == "true"
            pred_flag = str(row["is_main_pdt_pred"]).lower() == "true"

            if gt_flag and pred_flag:
                if product_match(row["product_gt"], row["product_pred"]):
                    correct_any = True
                    break

        if correct_any:
            TP += 1
        else:
            if pred_any:
                FP += 1
            if gt_any:
                FN += 1

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# =========================================================
# MAIN
# =========================================================
def evaluate(input_file):
    df = pd.read_csv(input_file)

    print("\n===== STRICT (Row-level with overlap) =====")
    strict = strict_metrics(df)
    for k, v in strict.items():
        print(f"{k}: {v}")

    print("\n===== AT-LEAST-ONE CORRECT (Query-level with overlap) =====")
    relaxed = atleast_one_metrics(df)
    for k, v in relaxed.items():
        print(f"{k}: {v}")


# ---------- RUN ----------
if __name__ == "__main__":
    evaluate("input.csv")
