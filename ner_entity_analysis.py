import pandas as pd
import ast
import re


# ---------- تنظيف np.int64 ----------
def clean_numpy_ints(text):
    if pd.isna(text):
        return "[]"
    return re.sub(r'np\.int64\((\d+)\)', r'\1', text)


# ---------- Normalize spans ----------
def normalize_spans(spans):
    normalized = []
    for label, start, end in spans:
        normalized.append((label, int(start), int(end)))
    return normalized


# ---------- Extract text ----------
def get_span_text(text, start, end):
    return text[start:end]


# ---------- Overlap check ----------
def spans_overlap(s1, e1, s2, e2):
    return max(s1, s2) < min(e1, e2)


# ---------- Core evaluation ----------
def evaluate_row(query, gt_spans, pred_spans):
    results = []

    matched_gt = set()
    matched_pred = set()

    # ---- 1. Exact Match (TP) ----
    for i, (g_label, g_start, g_end) in enumerate(gt_spans):
        for j, (p_label, p_start, p_end) in enumerate(pred_spans):
            if i in matched_gt or j in matched_pred:
                continue

            if g_label == p_label and g_start == p_start and g_end == p_end:
                results.append({
                    "Query": query,
                    "entity": g_label,
                    "gt_value": get_span_text(query, g_start, g_end),
                    "pred_value": get_span_text(query, p_start, p_end),
                    "status": "TP"
                })
                matched_gt.add(i)
                matched_pred.add(j)

    # ---- 2. Partial Match ----
    for i, (g_label, g_start, g_end) in enumerate(gt_spans):
        for j, (p_label, p_start, p_end) in enumerate(pred_spans):
            if i in matched_gt or j in matched_pred:
                continue

            if g_label == p_label and spans_overlap(g_start, g_end, p_start, p_end):
                results.append({
                    "Query": query,
                    "entity": g_label,
                    "gt_value": get_span_text(query, g_start, g_end),
                    "pred_value": get_span_text(query, p_start, p_end),
                    "status": "PARTIAL"
                })
                matched_gt.add(i)
                matched_pred.add(j)

    # ---- 3. False Negatives ----
    for i, (g_label, g_start, g_end) in enumerate(gt_spans):
        if i not in matched_gt:
            results.append({
                "Query": query,
                "entity": g_label,
                "gt_value": get_span_text(query, g_start, g_end),
                "pred_value": "",
                "status": "FN"
            })

    # ---- 4. False Positives ----
    for j, (p_label, p_start, p_end) in enumerate(pred_spans):
        if j not in matched_pred:
            results.append({
                "Query": query,
                "entity": p_label,
                "gt_value": "",
                "pred_value": get_span_text(query, p_start, p_end),
                "status": "FP"
            })

    return results


# ---------- Main pipeline ----------
def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    all_results = []

    for _, row in df.iterrows():
        query = row["Query"]

        # Clean np.int64 → int
        gt_text = clean_numpy_ints(row.get("true", "[]"))
        pred_text = clean_numpy_ints(row.get("pred", "[]"))

        # Parse safely
        try:
            gt_spans = normalize_spans(ast.literal_eval(gt_text))
        except:
            gt_spans = []

        try:
            pred_spans = normalize_spans(ast.literal_eval(pred_text))
        except:
            pred_spans = []

        row_results = evaluate_row(query, gt_spans, pred_spans)
        all_results.extend(row_results)

    result_df = pd.DataFrame(all_results)

    # Save
    result_df.to_csv(output_file, index=False)

    # Summary
    print("\n=== Evaluation Summary ===")
    print(result_df["status"].value_counts())


# ---------- Run ----------
if __name__ == "__main__":
    process_csv("input.csv", "output_analysis.csv")
