import ast
import csv
import math

# --------------------------------------------------------
# Safe loader (handles malformed GT or missing columns)
# --------------------------------------------------------
def safe_load_list(s, query=""):
    try:
        lst = ast.literal_eval(s)
        if isinstance(lst, list):
            return lst
        print(f"[WARN] GT/PRED not list for query: {query}")
        return []
    except Exception:
        print(f"[ERROR] Failed parsing GT/PRED for query: {query}")
        return []

# --------------------------------------------------------
# Flexible span overlap check
# --------------------------------------------------------
def flexible_span_overlap(gt, pred, tolerance=2, min_overlap_ratio=0.3):
    """
    gt = [text, start, end, adj, adp, is_main]
    pred = [text, start, end, adj, adp, is_main, label]
    Returns True if spans match with tolerance.
    """

    try:
        gs, ge = int(gt[1]), int(gt[2])
        ps, pe = int(pred[1]), int(pred[2])
    except:
        return False

    # soft boundary tolerance
    if abs(gs - ps) <= tolerance and abs(ge - pe) <= tolerance:
        return True

    # compute overlap
    overlap = max(0, min(ge, pe) - max(gs, ps))
    gt_len = max(1, ge - gs)
    pred_len = max(1, pe - ps)

    ratio = overlap / min(gt_len, pred_len)

    return ratio >= min_overlap_ratio

# --------------------------------------------------------
# Evaluation logic:
# Only consider PRED where pred.is_main_product == True.
# Then:
#    If span matches a GT that has is_main_product == True → TP
#    Else → FP
#
# No FN counted (as requested).
# --------------------------------------------------------
def evaluate_row(query, gt_list, pred_list):
    results = []

    # clean malformed GT elements
    cleaned_gt = []
    for g in gt_list:
        if len(g) < 6:
            print(f"[WARN] GT element missing fields for query: {query}")
            continue
        cleaned_gt.append(g)

    # Filter preds: ONLY those with pred.is_main_product == True
    pred_main = []
    for p in pred_list:
        if len(p) < 6:
            continue
        try:
            if bool(p[5]) is True:
                pred_main.append(p)
        except:
            continue

    # For each pred_main evaluate TP / FP
    for p in pred_main:
        label = None
        if len(p) >= 7:
            label = p[6]

        # check correct label
        label_valid = label in {"product_name", "accessory"}

        # find matching GT by flexible span
        match_gt = None
        for g in cleaned_gt:
            if flexible_span_overlap(g, p):
                match_gt = g
                break

        if match_gt:
            if match_gt[5] is True and label_valid:
                status = "TP"
            else:
                status = "FP"
        else:
            # no GT span matches → FP
            status = "FP"

        results.append({
            "query": query,
            "gt_text": match_gt[0] if match_gt else "",
            "pred_text": p[0],
            "pred_label": label,
            "status": status
        })

    return results

# --------------------------------------------------------
# CSV Evaluation wrapper
# --------------------------------------------------------
def evaluate_csv(path, out_csv="eval_report.csv"):
    final_rows = []
    tp = 0
    fp = 0

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row.get("query", "")
            gt_list = safe_load_list(row.get("gt", "[]"), query)
            pred_list = safe_load_list(row.get("preds", "[]"), query)

            row_results = evaluate_row(query, gt_list, pred_list)
            final_rows.extend(row_results)

    # compute precision only
    for r in final_rows:
        if r["status"] == "TP":
            tp += 1
        elif r["status"] == "FP":
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # write report
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query", "gt_text", "pred_text", "pred_label", "status"
        ])
        writer.writeheader()
        writer.writerows(final_rows)

    return {
        "precision": precision,
        "tp": tp,
        "fp": fp,
        "total_pred_positive": tp + fp,
        "output_csv": out_csv
    }

# --------------------------------------------------------
# Main entry
# --------------------------------------------------------
if __name__ == "__main__":
    res = evaluate_csv("input.csv")
    print("\nFinal Evaluation:")
    print(res)
