import ast
import csv

VALID_LABELS = {"product_name", "accessory"}

def load_list(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        print("❌ Error parsing row:", s)
        return []

def spans_overlap(g1, g2):
    """
    g1 = (start, end), g2 = (start, end)
    return True if overlap exists
    """
    return not (g1[1] < g2[0] or g2[1] < g1[0])


def find_best_match(gt, pred_list):
    """
    Match using span overlap (not exact equality)
    If multiple overlap, choose first.
    """
    gs, ge = gt[1], gt[2]

    for p in pred_list:
        ps, pe = p[1], p[2]
        if spans_overlap((gs, ge), (ps, pe)):
            return p
    return None


def evaluate_label(gt_list, pred_list, query, report_rows):
    """
    Evaluate only GT items where is_main_product == True
    GT → requires a valid label
    """

    y_true = []
    y_pred = []

    for gt in gt_list:
        text, start, end, adj, adp, is_main = gt

        if not is_main:
            # ignore non-main-product in GT
            continue

        # GT expects a correct label
        y_true.append(1)

        matched_pred = find_best_match(gt, pred_list)

        if matched_pred:
            pred_label = matched_pred[-1] if len(matched_pred) >= 7 else None
        else:
            pred_label = None

        # classification
        if pred_label in VALID_LABELS:
            y_pred.append(1)
            status = "TP"
        else:
            y_pred.append(0)
            status = "FN"

        report_rows.append({
            "query": query,
            "gt_text": text,
            "gt_start": start,
            "gt_end": end,
            "gt_is_main": is_main,
            "pred_text": matched_pred[0] if matched_pred else "",
            "pred_label": pred_label,
            "status": status
        })

    # FP: predictions that assigned a valid label but no GT main-product matches
    for p in pred_list:
        pred_label = p[-1] if len(p) >= 7 else None

        if pred_label in VALID_LABELS:
            # Check if any GT-main-product overlaps this pred
            matched_gt = None
            for gt in gt_list:
                if gt[5] == True:
                    if spans_overlap((gt[1], gt[2]), (p[1], p[2])):
                        matched_gt = gt
                        break

            if not matched_gt:
                # FP
                y_true.append(0)
                y_pred.append(1)
                report_rows.append({
                    "query": query,
                    "gt_text": "",
                    "gt_start": "",
                    "gt_end": "",
                    "gt_is_main": "",
                    "pred_text": p[0],
                    "pred_label": pred_label,
                    "status": "FP"
                })

    return y_true, y_pred


def evaluate_csv(path):
    all_true = []
    all_pred = []
    report_rows = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_list = load_list(row["gt"])
            pred_list = load_list(row["preds"])
            query = row.get("query", "")

            yt, yp = evaluate_label(gt_list, pred_list, query, report_rows)
            all_true.extend(yt)
            all_pred.extend(yp)

    # Save the report
    with open("label_eval_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "query",
            "gt_text", "gt_start", "gt_end", "gt_is_main",
            "pred_text", "pred_label",
            "status"
        ])
        w.writeheader()
        w.writerows(report_rows)

    # Global metrics
    TP = sum(1 for t, p in zip(all_true, all_pred) if t == 1 and p == 1)
    FN = sum(1 for t, p in zip(all_true, all_pred) if t == 1 and p == 0)
    FP = sum(1 for t, p in zip(all_true, all_pred) if t == 0 and p == 1)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": TP,
        "FP": FP,
        "FN": FN
    }


if __name__ == "__main__":
    result = evaluate_csv("input.csv")
    print(result)
    print("Report saved → label_eval_report.csv")
