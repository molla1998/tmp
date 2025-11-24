import ast
import csv
from sklearn.metrics import precision_recall_fscore_support

def load_list(s):
    return ast.literal_eval(s)

def extract_labels(gt_list, pred_list):
    """
    Evaluate label correctness only when gt.is_main_product == True.
    """
    y_true = []
    y_pred = []

    for gt in gt_list:
        gt_is_main = gt[5]

        # Find best matching pred by span overlap (simple matching)
        matched_pred = None
        for p in pred_list:
            if p[1] == gt[1] and p[2] == gt[2]:
                matched_pred = p
                break

        if gt_is_main:
            y_true.append(1)  # requires label

            if matched_pred and len(matched_pred) >= 7:
                y_pred.append(1)  # predicted label exists
            else:
                y_pred.append(0)  # missing label
        else:
            # GT does NOT require label
            # Check if prediction wrongfully assigns one
            y_true.append(0)
            if matched_pred and len(matched_pred) >= 7:
                y_pred.append(1)  # predicted label incorrectly
            else:
                y_pred.append(0)

    return y_true, y_pred


def extract_adj(gt_list, pred_list):
    """
    Evaluate adj_list correctness for all instances.
    """
    y_true = []
    y_pred = []

    for gt in gt_list:
        gt_adj = gt[3]

        # match pred by span
        matched_pred = None
        for p in pred_list:
            if p[1] == gt[1] and p[2] == gt[2]:
                matched_pred = p
                break

        pred_adj = matched_pred[3] if matched_pred else ""

        # treat adj as present or absent
        y_true.append(1 if gt_adj else 0)
        y_pred.append(1 if pred_adj else 0)

    return y_true, y_pred


def evaluate_csv(path):
    all_ytrue_label = []
    all_ypred_label = []

    all_ytrue_adj = []
    all_ypred_adj = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_list = load_list(row["gt"])
            pred_list = load_list(row["preds"])

            # LABEL evaluation
            yt_l, yp_l = extract_labels(gt_list, pred_list)
            all_ytrue_label.extend(yt_l)
            all_ypred_label.extend(yp_l)

            # ADJ evaluation
            yt_a, yp_a = extract_adj(gt_list, pred_list)
            all_ytrue_adj.extend(yt_a)
            all_ypred_adj.extend(yp_a)

    # Global metrics
    label_prec, label_rec, label_f1, _ = precision_recall_fscore_support(
        all_ytrue_label, all_ypred_label, average="binary"
    )

    adj_prec, adj_rec, adj_f1, _ = precision_recall_fscore_support(
        all_ytrue_adj, all_ypred_adj, average="binary"
    )

    return {
        "precision_label": label_prec,
        "recall_label": label_rec,
        "f1_label": label_f1,
        "precision_adj": adj_prec,
        "recall_adj": adj_rec,
        "f1_adj": adj_f1,
    }


if __name__ == "__main__":
    res = evaluate_csv("input.csv")
    print(res)
