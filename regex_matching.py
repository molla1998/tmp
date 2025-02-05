import pandas as pd
import re

def longest_common_prefix_regex(strings):
    """Finds the longest common prefix and returns it as a regex pattern."""
    if not strings:
        return ""

    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""

    # Escape special regex characters and add .* for flexible matching
    return re.escape(prefix) + ".*"

def longest_common_suffix_regex(strings):
    """Finds the longest common suffix and returns it as a regex pattern."""
    if not strings:
        return ""

    suffix = strings[0]
    for s in strings[1:]:
        while not s.endswith(suffix):
            suffix = suffix[1:]
            if not suffix:
                return ""

    # Escape special regex characters and add .* for flexible matching
    return ".*" + re.escape(suffix)

# Load CSV file
input_csv = "products.csv"  # Update with your file name
df = pd.read_csv(input_csv)  # Assuming columns: "Product ID", "Category"

# Group by category and calculate longest prefix & suffix regex
results = []
for category, group in df.groupby("Category"):
    product_ids = group["Product ID"].astype(str).tolist()
    
    common_prefix_regex = longest_common_prefix_regex(product_ids)
    common_suffix_regex = longest_common_suffix_regex(product_ids)

    results.append({
        "Category": category,
        "Prefix Regex": common_prefix_regex,
        "Suffix Regex": common_suffix_regex
    })

# Convert to DataFrame
output_df = pd.DataFrame(results)

# Save output CSV
output_csv = "category_prefix_suffix_regex.csv"
output_df.to_csv(output_csv, index=False)

print(f"Output saved to {output_csv}")
