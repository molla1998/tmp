import csv
import ast

VALID_LABELS = {"product_name", "accessory"}

def safe_eval(s, query):
    """Safely parse list, and if failing, print the query causing error."""
    try:
        return ast.literal_eval(s)
    except Exception:
        print(f"[ERROR] Failed to parse GT/PRED for query:\n{query}\nValue: {s}")
        return []


def span_overlap(a_start, a_end, b_start, b_end):
    """Returns True if two spans overlap."""
    return not (a_end < b_start or b_end < a_start)


def evaluate(input_csv, output_csv="eval_report.csv"):
    rows_out = []
    TP = 0
    FP = 0

    with open(input_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            query = row["query"]
            gt_list = safe_eval(row["gt"], query)
            pred_list = safe_eval(row["preds"], query)

            # Normalize GT entries (may be malformed)
            cleaned_gt = []
            for g in gt_list:
                try:
                    text = g[0]
                    start = g[1]
                    end = g[2]
                    adj = g[3] if len(g) > 3 else ""
                    adp = g[4] if len(g) > 4 else ""
                    is_main = g[5] if len(g) > 5 else False
                    cleaned_gt.append([text, start, end, adj, adp, is_main])
                except Exception:
                    print(f"[WARNING] Skipping malformed GT row in query: {query}")

            # Evaluate only pred.is_main_product == TRUE
            for p in pred_list:
                try:
                    pred_text = p[0]
                    ps, pe = p[1], p[2]
                    pred_is_main = p[5] if len(p) > 5 else False
                    pred_label = p[6] if len(p) > 6 else None
                except Exception:
                    print(f"[WARNING] Skipping malformed PRED in query: {query}")
                    continue

                if pred_is_main is not True:
                    continue  # ignore all preds not marked as main

                if pred_label not in VALID_LABELS:
                    continue  # ignore preds not having valid label

                # Find GT match by overlap
                matched_gt = None
                for g in cleaned_gt:
                    if span_overlap(ps, pe, g[1], g[2]):
                        matched_gt = g
                        break

                if matched_gt:
                    gt_text = matched_gt[0]
                    if matched_gt[5] is True:
                        status = "TP"
                        TP += 1
                    else:
                        status = "FP"
                        FP += 1
                else:
                    status = "FP"
                    gt_text = ""

                    FP += 1

                rows_out.append([query, pred_text, gt_text, status])

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Write report CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "pred_text", "gt_text", "status"])
        w.writerows(rows_out)

    print("==== FINAL PRECISION ====")
    print("Precision:", precision)
    print("TP:", TP)
    print("FP:", FP)
    print("=========================")

    return precision
