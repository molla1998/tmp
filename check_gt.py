import csv
import ast

VALID_LABELS = {"product_name", "accessory"}


def safe_load(s, query=""):
    """Load list from CSV safely, print query if broken."""
    try:
        return ast.literal_eval(s)
    except Exception:
        print(f"[ERROR] Could not parse GT/PRED for query: {query}")
        return []


def span_overlap(a_start, a_end, b_start, b_end):
    """Return True if spans overlap."""
    return not (a_end < b_start or b_end < a_start)


def match_pred(gt_item, pred_list):
    """Return best pred that overlaps with GT span."""
    g_start, g_end = gt_item[1], gt_item[2]

    for p in pred_list:
        if span_overlap(g_start, g_end, p[1], p[2]):
            return p

    return None


def evaluate_row(gt_list, pred_list):
    """
    Evaluate only GT entries where is_main_product == True.
    Return list of records used for final CSV reporting.
    """
    records = []

    # Filter only main-product GTs
    gt_main = [g for g in gt_list if g[5] == True]

    # Collect preds that are main-product to detect FP later
    pred_main = [p for p in pred_list if p[5] == True]

    matched_pred_ids = set()

    for gt in gt_main:
        gt_text = gt[0]
        matched = match_pred(gt, pred_list)

        if matched:
            matched_pred_ids.add(id(matched))
            pred_text = matched[0]
            pred_label = matched[-1] if len(matched) >= 7 else None

            if matched[5] == True and pred_label in VALID_LABELS:
                status = "TP"
            else:
                status = "FN"
        else:
            pred_text = ""
            pred_label = None
            status = "FN"

        records.append({
            "gt_text": gt_text,
            "pred_text": pred_text,
            "pred_label": pred_label,
            "status": status
        })

    # Detect FP: any pred-main-product not matched to any GT-main-product
    for p in pred_main:
        if id(p) not in matched_pred_ids:
            pred_text = p[0]
            pred_label = p[-1] if len(p) >= 7 else None

            records.append({
                "gt_text": "",
                "pred_text": pred_text,
                "pred_label": pred_label,
                "status": "FP"
            })

    return records


def evaluate_csv(input_path, output_path="eval_report.csv"):
    all_records = []

    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            gt_list = safe_load(row["gt"], row.get("query", ""))
            pred_list = safe_load(row["preds"], row.get("query", ""))

            recs = evaluate_row(gt_list, pred_list)
            for r in recs:
                r["query"] = row["query"]
            all_records.extend(recs)

    # Write report
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query", "gt_text", "pred_text", "pred_label", "status"]
        )
        writer.writeheader()
        writer.writerows(all_records)

    # Compute global TP/FP/FN
    TP = sum(1 for r in all_records if r["status"] == "TP")
    FP = sum(1 for r in all_records if r["status"] == "FP")
    FN = sum(1 for r in all_records if r["status"] == "FN")

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0

    print("\n=== FINAL METRICS ===")
    print(f"TP = {TP}")
    print(f"FP = {FP}")
    print(f"FN = {FN}")
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1        = {f1:.4f}")
    print("\nCSV report generated:", output_path)


if __name__ == "__main__":
    evaluate_csv("input.csv")
