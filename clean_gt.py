import csv
import ast

def load_list(s):
    """Safely load list from string."""
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

def clean_gt(gt_list):
    """Keep only entries where is_main_product == True."""
    cleaned = []
    for item in gt_list:
        if len(item) >= 6 and item[5] is True:
            cleaned.append(item)
    return cleaned

def process_csv(input_path, output_path):
    with open(input_path, encoding="utf-8") as infile, \
         open(output_path, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            gt_list = load_list(row["gt"])
            cleaned = clean_gt(gt_list)
            row["gt"] = str(cleaned)
            writer.writerow(row)

    print(f"GT cleaned and saved to {output_path}")

if __name__ == "__main__":
    process_csv("input.csv", "cleaned_gt.csv")
