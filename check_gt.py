import csv
import ast
import pandas as pd

def load_list_safe(s, query=""):
    try:
        return ast.literal_eval(s)
    except Exception:
        print("AST parse error for query:", query)
        return []

def span_overlap(a_start, a_end, b_start, b_end):
    return max(0, min(a_end, b_end) - max(a_start, b_start)) > 0

def evaluate_main_product(input_csv):
    tp = fp = fn = 0
    rows_report = []

    with open(input_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            gt = load_list_safe(row["gt"], row.get("query", ""))
            pred = load_list_safe(row["preds"], row.get("query", ""))

            # Keep only preds where is_main_product == True
            pred_main = [p for p in pred if p[5] is True]

            # Track GT true items
            gt_main = [g for g in gt if g[5] is True]

            matched_gt_ids = set()

            # Process each main prediction
            for p in pred_main:
                p_start, p_end = p[1], p[2]
                matched_gt = None

                # Find overlapping GT
                for idx, g in enumerate(gt):
                    if span_overlap(p_start, p_end, g[1], g[2]):
                        matched_gt = (idx, g)
                        break

                if matched_gt:
                    idx, g = matched_gt
                    matched_gt_ids.add(idx)

                    if g[5] is True:
                        tp += 1
                        rows_report.append([g[0], p[0], "TP"])
                    else:
                        fp += 1
                        rows_report.append([g[0], p[0], "FP"])
                else:
                    # pred main product but no GT overlap
                    fp += 1
                    rows_report.append(["", p[0], "FP"])

            # FN = GT true but no predicted main product overlapped
            for idx, g in enumerate(gt_main):
                # If this GT index was never matched → FN
                if idx not in matched_gt_ids:
                    fn += 1
                    rows_report.append([g[0], "", "FN"])

    # compute metrics
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    # save report
    df = pd.DataFrame(rows_report, columns=["gt_text", "pred_text", "status"])
    df.to_csv("eval_report.csv", index=False)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


if __name__ == "__main__":
    result = evaluate_main_product("input.csv")
    print(result)
    print("Report saved to eval_report.csv")
