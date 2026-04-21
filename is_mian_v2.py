import pandas as pd


# ---------- Normalize ----------
def normalize(text):
    return str(text).lower().strip()


# ---------- Overlap match ----------
def product_match(p1, p2):
    p1 = normalize(p1)
    p2 = normalize(p2)

    if p1 in p2 or p2 in p1:
        return True

    t1 = set(p1.split())
    t2 = set(p2.split())

    return len(t1 & t2) > 0


# =========================================================
# 1️⃣ STRICT (Row-level with separated PDT errors)
# =========================================================
def strict_with_analysis(df, output_file):
    rows = []

    TP = FP = FN = TN = 0
    FP_PDT = FN_PDT = 0

    for _, row in df.iterrows():
        query = row["query"]
        gt_product = row["product_gt"]
        pred_product = row["product_pred"]

        gt_flag = str(row["is_main_pdt_gt"]).lower() == "true"
        pred_flag = str(row["is_main_pdt_pred"]).lower() == "true"

        match = product_match(gt_product, pred_product)

        # ---- status logic ----
        if gt_flag and pred_flag:
            if match:
                status = "TP"
                TP += 1
            else:
                status = "FN_PDT"
                FN_PDT += 1

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
            "product_gt": gt_product,
            "product_pred": pred_product,
            "is_main_pdt_gt": gt_flag,
            "is_main_pdt_pred": pred_flag,
            "match": match,
            "status": status
        })

    # ---- Save analysis ----
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_file, index=False)

    # ---- Metrics (ONLY pure classification errors) ----
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    metrics = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "FP_PDT": FP_PDT,
        "FN_PDT": FN_PDT,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    return metrics


# =========================================================
# 2️⃣ QUERY-LEVEL (ignore PDT mismatch for correctness)
# =========================================================
def atleast_one_with_analysis(df, output_file):
    rows = []
    TP = FP = FN = 0

    grouped = df.groupby("query")

    for query, group in grouped:
        gt_any = any(str(x).lower() == "true" for x in group["is_main_pdt_gt"])
        pred_any = any(str(x).lower() == "true" for x in group["is_main_pdt_pred"])

        # 👉 only check flags, ignore product mismatch
        correct_any = any(
            (str(gt).lower() == "true") and (str(pred).lower() == "true")
            for gt, pred in zip(group["is_main_pdt_gt"], group["is_main_pdt_pred"])
        )

        if correct_any:
            status = "TP"
            TP += 1
        else:
            if pred_any:
                status = "FP"
                FP += 1
            elif gt_any:
                status = "FN"
                FN += 1
            else:
                status = "TN"

        rows.append({
            "query": query,
            "gt_has_main": gt_any,
            "pred_has_main": pred_any,
            "atleast_one_flag_correct": correct_any,
            "status": status
        })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_file, index=False)

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    metrics = {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    return metrics


# =========================================================
# MAIN
# =========================================================
def evaluate(input_file):
    df = pd.read_csv(input_file)

    print("\n===== STRICT (Row-level, clean metrics) =====")
    strict_metrics = strict_with_analysis(df, "strict_analysis.csv")
    for k, v in strict_metrics.items():
        print(f"{k}: {v}")

    print("\n===== AT-LEAST-ONE (Query-level, flag only) =====")
    relaxed_metrics = atleast_one_with_analysis(df, "query_analysis.csv")
    for k, v in relaxed_metrics.items():
        print(f"{k}: {v}")


# ---------- RUN ----------
if __name__ == "__main__":
    evaluate("input.csv")
