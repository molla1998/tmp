import re
from collections import defaultdict

def tokenize(text):
    return re.findall(r"[a-zA-Z0-9\-]+", text.lower())

def find_adj_for_main_products(query, main_products, adjectives, window=3):
    tokens = tokenize(query)
    adj_set = set(a.lower() for a in adjectives)

    results = defaultdict(list)

    for product in main_products:
        prod_tokens = tokenize(product)

        # find the product phrase in the query tokens
        for i in range(len(tokens) - len(prod_tokens) + 1):
            if tokens[i:i+len(prod_tokens)] == prod_tokens:
                p_start = i  # use FIRST token index as anchor (your rule)

                # only look to the LEFT: p_start-1 ... p_start-window
                for k in range(1, window + 1):
                    idx = p_start - k
                    if idx < 0:
                        break
                    if tokens[idx] in adj_set:
                        results[product].append(tokens[idx])

    return dict(results)

# -------------------
# Demo
# -------------------
if __name__ == "__main__":
    query = "best premium galaxy s24 with 8gb ram"
    main_products = ["galaxy s24"]
    adjectives = ["best", "premium"]

    adj_map = find_adj_for_main_products(query, main_products, adjectives, window=3)
    print(adj_map)
