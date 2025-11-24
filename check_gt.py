import ast
import csv

def load_list_safe(s, query):
    try:
        return ast.literal_eval(s)
    except Exception:
        print("❌ Error parsing row for query:", query)
        return []

def span_overlap(a_start, a_end, b_start, b_end):
    """Check if two spans overlap."""
    return not (a_end < b_start or b_end < a_start)

def evaluate_main_product(gt_list, pred_list):
    """Evaluate TP/FP/FN exactly as per your final rule."""

    # 1. Filter preds: only those where pred.is_main_product == True
    pred_true = [p for p in pred_list if p[5] is True]

    # Track results
    results = []  # each row: dict(text_gt, text_pred, status)
    used_gt = set()

    # --- Evaluate TP / FP ---
    for p in pred_true:
        p_text, p_s, p_e = p[0], p[1], p[2]

        matched_gt = None
        for i, g in enumerate(gt_list):
            g_text, g_s, g_e, _, _, g_main = g
            if span_overlap(p_s, p_e, g_s, g_e):
                matched_gt = (i, g)
                break

        if matched_gt:
            idx, g = matched_gt
            used_gt.add(idx)

            if g[5] is True:
                status = "TP"
            else:
                status = "FP"

        else:
            status = "FP"  # predicted main product that does not exist in GT

        results.append({
            "gt_text": g[0] if matched_gt else "",
            "pred_text": p_text,
            "status": status
        })

    # --- Evaluate FN (GT main items not predicted) ---
    for i, g in enumerate(gt_list):
        if g[5] is True and i not in used_gt:
            results.append({
                "gt_text": g[0],
                "pred_text": "",
                "status": "FN"
            })

    return results


def evaluate_csv(path):
    all_rows = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            gt_list = load_list_safe(row["gt"], row["query"])
            pred_list = load_list_safe(row["preds"], row["query"])

            eval_rows = evaluate_main_product(gt_list, pred_list)
            for r in eval_rows:
                r["query"] = row["query"]
                all_rows.append(r)

    # write output
    out = "eval_report.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "gt_text", "pred_text", "status"])
        writer.writeheader()
        writer.writerows(all_rows)

    print("✔ Evaluation saved to", out)


if __name__ == "__main__":
    evaluate_csv("input.csv")
