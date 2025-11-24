import csv
import ast

def load_list_safe(s, query):
    try:
        return ast.literal_eval(s)
    except Exception:
        print(f"[ERROR parsing GT/PRED] Query: {query}")
        return []

def spans_overlap(a_start, a_end, b_start, b_end):
    """Return True if two spans overlap."""
    return not (a_end < b_start or b_end < a_start)

def evaluate_csv(path):
    TP = 0
    FP = 0
    report_rows = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            query = row["query"]
            gt_list = load_list_safe(row["gt"], query)
            pred_list = load_list_safe(row["preds"], query)

            # Only consider preds where pred.is_main_product == True
            filtered_preds = [p for p in pred_list if len(p) >= 6 and p[5] == True]

            for pred in filtered_preds:
                p_text, p_start, p_end = pred[0], pred[1], pred[2]
                pred_label = pred[6] if len(pred) >= 7 else None

                # Must belong to {product_name, accessory}
                if pred_label not in {"product_name", "accessory"}:
                    # Label not valid → treat as FP
                    FP += 1
                    report_rows.append([query, p_text, "", "fp"])
                    continue

                matched_gt = None
                for gt in gt_list:
                    if spans_overlap(p_start, p_end, gt[1], gt[2]):
                        matched_gt = gt
                        break

                if matched_gt:
                    gt_is_main = matched_gt[5]
                    gt_text = matched_gt[0]

                    if gt_is_main == True:
                        TP += 1
                        status = "tp"
                    else:
                        FP += 1
                        status = "fp"

                    report_rows.append([query, p_text, gt_text, status])

                else:
                    # No overlapping GT → FP
                    FP += 1
                    report_rows.append([query, p_text, "", "fp"])

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Write the report
    with open("eval_report.csv", "w", newline="", encoding="utf-8") as w:
        writer = csv.writer(w)
        writer.writerow(["query", "pred_text", "gt_text", "status"])
        writer.writerows(report_rows)

    return {
        "TP": TP,
        "FP": FP,
        "precision": precision,
        "report_file": "eval_report.csv"
    }



if __name__ == "__main__":
    result = evaluate_csv("input.csv")
    print(result)
