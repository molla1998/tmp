import csv
import ast

def safe_load_list(s, query):
    """Safely load list using literal_eval; print query if parsing fails."""
    try:
        data = ast.literal_eval(s)
        # Fix missing columns in GT or Pred rows
        cleaned = []
        for item in data:
            # GT format: [text, start, end, adj, adp, is_main]
            # PRED format: [text, start, end, adj, adp, is_main, label]
            if len(item) < 6:  
                item = item + [None] * (6 - len(item))
            cleaned.append(item)
        return cleaned
    except Exception:
        print(f"❌ Parse error for query: {query}")
        return []

def span_overlap(a_start, a_end, b_start, b_end):
    """Return True if spans overlap."""
    return not (a_end < b_start or b_end < a_start)

def evaluate(input_csv, output_csv):
    rows = []
    TP = 0
    FP = 0

    with open(input_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            query = row["query"]
            gt_list = safe_load_list(row["gt"], query)
            pred_list = safe_load_list(row["preds"], query)

            # ---- Only consider preds with is_main_product == True ----
            main_preds = [p for p in pred_list if len(p) >= 6 and p[5] == True]

            for pred in main_preds:
                p_text, p_start, p_end = pred[0], pred[1], pred[2]

                matched_gt = None

                # Find ANY overlapping GT span
                for gt in gt_list:
                    if span_overlap(p_start, p_end, gt[1], gt[2]):
                        matched_gt = gt
                        break

                if matched_gt:
                    gt_is_main = matched_gt[5]  # True or False

                    if gt_is_main:  # GT True → TP
                        status = "TP"
                        TP += 1
                    else:  # GT False → FP
                        status = "FP"
                        FP += 1
                else:
                    # No GT span matched → FP
                    status = "FP"
                    FP += 1
                    matched_gt = ["<no match>", None, None, None, None, False]

                rows.append({
                    "query": query,
                    "pred_text": p_text,
                    "gt_text": matched_gt[0],
                    "status": status
                })

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # ---- Write Report CSV ----
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "pred_text", "gt_text", "status"])
        writer.writeheader()
        writer.writerows(rows)

    print("========== FINAL RESULTS ==========")
    print(f"TP = {TP}")
    print(f"FP = {FP}")
    print(f"Precision = {precision:.4f}")
    print("Report saved to:", output_csv)


# Run
if __name__ == "__main__":
    evaluate("input.csv", "eval_report.csv")
