import csv
import ast
import pandas as pd

def safe_load_list(s, query=""):
    """Safely parse list; if error, print query."""
    try:
        return ast.literal_eval(s)
    except Exception:
        print("❌ Failed to parse GT/PRED for query:", query)
        return []

def spans_overlap(a_start, a_end, b_start, b_end):
    return not (a_end < b_start or b_end < a_start)

def evaluate_row(query, gt_list, pred_list):
    # Filter only PRED items where is_main_product == True
    pred_main = [p for p in pred_list if len(p) >= 6 and p[5] == True]

    # GT main items
    gt_main = [g for g in gt_list if g[5] == True]

    results = []

    used_gt = set()   # for tracking FN later

    # Evaluate preds
    for p in pred_main:
        p_text, ps, pe = p[0], p[1], p[2]

        matched_gt = None
        for i, g in enumerate(gt_main):
            if spans_overlap(ps, pe, g[1], g[2]):
                matched_gt = (i, g)
                used_gt.add(i)
                break

        if matched_gt:
            status = "TP"
        else:
            status = "FP"

        results.append({
            "query": query,
            "gt_text": matched_gt[1][0] if matched_gt else "",
            "pred_text": p_text,
            "gt_is_main": True if matched_gt else "",
            "pred_is_main": True,
            "status": status
        })

    # Compute FN: GT.main not matched by any pred.main
    for i, g in enumerate(gt_main):
        if i not in used_gt:
            results.append({
                "query": query,
                "gt_text": g[0],
                "pred_text": "",
                "gt_is_main": True,
                "pred_is_main": "",
                "status": "FN"
            })

    return results


def evaluate_csv(input_path, output_path="eval_report.csv"):
    final_rows = []

    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row["query"]
            gt = safe_load_list(row["gt"], query=query)
            pred = safe_load_list(row["preds"], query=query)

            row_eval = evaluate_row(query, gt, pred)
            final_rows.extend(row_eval)

    df = pd.DataFrame(final_rows)
    df.to_csv(output_path, index=False)
    print("✅ Report generated:", output_path)

    # Also print evaluation matrix
    tp = sum(df["status"] == "TP")
    fp = sum(df["status"] == "FP")
    fn = sum(df["status"] == "FN")

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print("\n📌 **Evaluation Matrix (Only pred.is_main_product == True)**")
    print(f"TP = {tp}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print("------------------------")
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1-score  = {f1:.4f}")

    return df


if __name__ == "__main__":
    evaluate_csv("input.csv", "eval_report.csv")
