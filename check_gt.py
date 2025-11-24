import ast
import csv

# ==============================
# Safe loader that handles errors
# ==============================
def safe_load_list(s, query=""):
    try:
        return ast.literal_eval(s)
    except Exception:
        print(f"[ERROR] Could not parse GT/PRED list for query:\n{query}\nRaw: {s}\n")
        return []

# ==============================
# More flexible span overlap
# Allows small drift (±2 chars default)
# ==============================
def span_overlap(a_start, a_end, b_start, b_end, tolerance=2):
    """
    Returns True if spans approximately overlap with tolerance.
    Good for Korean cases where token boundaries drift by 1–2 chars.
    """
    return not (a_end < b_start - tolerance or b_end < a_start - tolerance)

# ==============================
# Normalize GT/PRED element
# ==============================
def normalize_item(item):
    """
    Expected format for GT:
        [text, start, end, adj, adp, is_main_product]

    Expected format for PRED:
        [text, start, end, adj, adp, is_main_product, label]

    Some GT/PRED entries may be missing a field — handle safely.
    """
    text = item[0] if len(item) > 0 else ""
    start = item[1] if len(item) > 1 else -1
    end = item[2] if len(item) > 2 else -1
    adj = item[3] if len(item) > 3 else ""
    adp = item[4] if len(item) > 4 else ""
    is_main = item[5] if len(item) > 5 else False
    label = item[6] if len(item) > 6 else None  # Only for preds
    return text, start, end, adj, adp, is_main, label

# ==============================
# Evaluation logic
# ==============================
def evaluate_csv(input_path, output_path="report.csv"):
    rows_out = []
    tp = 0
    fp = 0

    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            query = row["query"]
            gt_list = safe_load_list(row["gt"], query)
            pred_list = safe_load_list(row["preds"], query)

            # Normalize all GT
            gt_norm = [normalize_item(g) for g in gt_list]

            # Only consider pred.is_main_product == True
            for p in pred_list:
                p_text, p_start, p_end, _, _, p_is_main, p_label = normalize_item(p)

                if not p_is_main:
                    continue  # Ignore completely

                # Check for matching GT span
                matched_gt = None
                for g in gt_norm:
                    g_text, g_start, g_end, _, _, g_is_main, _ = g

                    if span_overlap(p_start, p_end, g_start, g_end):
                        matched_gt = g
                        break

                # Evaluate TP/FP
                if matched_gt:
                    g_text, g_start, g_end, _, _, g_is_main, _ = matched_gt
                    if g_is_main:
                        status = "TP"
                        tp += 1
                    else:
                        status = "FP"
                        fp += 1
                else:
                    status = "FP"
                    g_text = ""   # no GT match
                    fp += 1

                rows_out.append({
                    "query": query,
                    "pred_text": p_text,
                    "gt_text": g_text,
                    "status": status
                })

    # Precision calculation
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Write report CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "pred_text", "gt_text", "status"])
        writer.writeheader()
        writer.writerows(rows_out)

    print("==== FINAL PRECISION ====")
    print(f"Precision: {precision:.4f}")
    print("=========================")

    print(f"Report saved to: {output_path}")

    return precision
            

# Run script
if __name__ == "__main__":
    evaluate_csv("input.csv", "report.csv")
