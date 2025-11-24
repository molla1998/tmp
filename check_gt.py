import csv
import ast

def safe_load_list(s, query):
    """
    Try to parse the GT list using ast.literal_eval.
    If parsing fails, print the query and the raw GT string.
    """
    try:
        return ast.literal_eval(s)
    except Exception as e:
        print("\n❌ ERROR parsing GT for query:")
        print("Query:", query)
        print("GT string:", s)
        print("Python Error:", e)
        return None  # or return [] if you prefer


def load_input_csv(path):
    rows = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            query = row.get("query", "")
            gt_raw = row.get("gt", "")

            gt = safe_load_list(gt_raw, query)
            rows.append({
                "query": query,
                "gt": gt,
                "preds_raw": row.get("preds", "")
            })

    return rows


if __name__ == "__main__":
    rows = load_input_csv("input.csv")

    print("\nLoaded rows:")
    for r in rows:
        print(r)
