import ast
import pandas as pd
from collections import defaultdict


def parse_annotation(raw):
    """
    Convert string representation of list into python list.
    """
    if isinstance(raw, list):
        return raw
    return ast.literal_eval(raw)


def overlap(a_s, a_e, b_s, b_e):
    """True if spans overlap."""
    return not (a_e < b_s or b_e < a_s)


def eval_custom_ner(gold, pred):
    """
    gold and pred: lists of annotations in the format:
    [text, start, end, adj_list, adp_list, label, is_main_product]

    pred format:
    [text, start, end, label, adj_list, adp_list, label, is_main_product]

    NOTE: adp is ignored for evaluation.
    """

    # ──────────────────────────────────────────────
    # METRIC BUCKETS
    # ──────────────────────────────────────────────
    metrics = {
        "product": {"tp": 0, "fp": 0, "fn": 0},
        "adj": {"tp": 0, "fp": 0, "fn": 0},
        "main_product": {"tp": 0, "fp": 0, "fn": 0}
    }

    # ──────────────────────────────────────────────
    # CONVERT pred TO UNIFIED FORMAT
    # pred comes as: [text, start, end, label (4th?), adj_list, adp_list, label, is_main]
    # We pick: text, start, end, adj_list, label, is_main
    # ──────────────────────────────────────────────
    processed_pred = []
    for p in pred:
        p_text = p[0]
        p_s = p[1]
        p_e = p[2]
        p_label = p[3]            # main label
        p_adj = p[4]
        p_is_main = p[7] if len(p) > 7 else False

        processed_pred.append([p_text, p_s, p_e, p_adj, p_label, p_is_main])


    # ──────────────────────────────────────────────
    # ONLY GT ENTRIES WHERE is_main_product = True
    # ──────────────────────────────────────────────
    gold_main = [g for g in gold if g[6] is True]

    used_pred_idx = set()

    # ──────────────────────────────────────────────
    # MATCH GT MAIN PRODUCT TO PRED MAIN CANDIDATES
    # ──────────────────────────────────────────────
    for g in gold_main:
        g_text, g_s, g_e, g_adj, g_adp, g_label, g_main = g

        g_adj_set = set([x.strip() for x in g_adj.split(",") if x.strip()]) if g_adj else set()

        found_match = False

        for i, p in enumerate(processed_pred):
            if i in used_pred_idx:
                continue

            p_text, p_s, p_e, p_adj, p_label, p_main = p

            # label must be product_name or accessory
            if p_label not in ("product_name", "accessory"):
                continue

            # spans must overlap
            if not overlap(g_s, g_e, p_s, p_e):
                continue

            # MAIN PRODUCT must match if GT is main-product
            if p_main is not True:
                continue

            # ADJ evaluation
            p_adj_set = set([x.strip() for x in p_adj.split(",") if x.strip()]) if p_adj else set()

            if not g_adj_set and not p_adj_set:
                # both empty → TP for adj
                metrics["adj"]["tp"] += 1
            else:
                # count adj TP/FN/FP tokenwise
                for a in g_adj_set:
                    if a in p_adj_set:
                        metrics["adj"]["tp"] += 1
                    else:
                        metrics["adj"]["fn"] += 1

                for a in p_adj_set:
                    if a not in g_adj_set:
                        metrics["adj"]["fp"] += 1

            # matched fully
            metrics["product"]["tp"] += 1
            metrics["main_product"]["tp"] += 1
            used_pred_idx.add(i)
            found_match = True
            break

        if not found_match:
            # FN for product + FN for main_product
            metrics["product"]["fn"] += 1
            metrics["main_product"]["fn"] += 1

            # adj FN for all gold adj items
            for a in g_adj_set:
                metrics["adj"]["fn"] += 1


    # ──────────────────────────────────────────────
    # COUNT FP: preds that were not matched but claim main_product
    # ──────────────────────────────────────────────
    for i, p in enumerate(processed_pred):
        if i not in used_pred_idx and p[5] is True:
            metrics["product"]["fp"] += 1
            metrics["main_product"]["fp"] += 1

            # adj fp
            p_adj = p[3]
            if p_adj:
                for a in p_adj.split(","):
                    a = a.strip()
                    if a:
                        metrics["adj"]["fp"] += 1


    # ──────────────────────────────────────────────
    # Compute final precision/recall/F1
    # ──────────────────────────────────────────────
    final = {}
    for name, m in metrics.items():
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        final[name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }

    return final



# ──────────────────────────────────────────────
# MAIN FUNCTION TO LOAD CSV
# ──────────────────────────────────────────────

def evaluate_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    all_results = []

    for idx, row in df.iterrows():
        gold = parse_annotation(row["gt"])
        pred = parse_annotation(row["pred"])

        score = eval_custom_ner(gold, pred)
        all_results.append(score)

    return all_results
