import json
import time
import torch
import networkx as nx
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


############################################
# CONFIG
############################################

THRESHOLD_MAP = {
    "memory": 0.40,
    "color": 0.30,
    "accessory": 0.40,
    "brand": 0.35,
    "product_product": 0.30
}

DEFAULT_THRESHOLD = 0.40


############################################
# LOAD MODEL
############################################

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)


############################################
# SAFE COSINE
############################################

def safe_cosine(a, b):

    a = np.array(a)
    b = np.array(b)

    if np.isnan(a).any() or np.isnan(b).any():
        return 0

    return cosine_similarity([a], [b])[0][0]


############################################
# SPAN EMBEDDINGS
############################################

def generate_span_embeddings(text, entities):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True
    )

    offsets = inputs["offset_mapping"][0]
    inputs.pop("offset_mapping")

    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state[0]

    for ent in entities:

        token_ids = []

        for i, (start, end) in enumerate(offsets):

            if start >= ent["start"] and end <= ent["end"]:
                token_ids.append(i)

        if not token_ids:
            ent["embedding"] = np.zeros(token_embeddings.shape[1])
        else:
            vec = token_embeddings[token_ids].mean(dim=0)
            ent["embedding"] = vec.numpy()

    return entities


############################################
# DISTANCE
############################################

def span_distance(a, b):

    return max(
        0,
        max(a["start"] - b["end"], b["start"] - a["end"])
    )


def distance_score(a, b):

    d = span_distance(a, b)

    return 1 / (d + 1)


############################################
# BUILD GRAPH
############################################

def build_graph(entities):

    G = nx.DiGraph()

    attributes = []
    products = []

    for e in entities:

        G.add_node(
            e["text"],
            label=e["label"],
            start=e["start"],
            end=e["end"]
        )

        if e["label"] == "product_name":
            products.append(e)
        else:
            attributes.append(e)

    ########################################
    # ATTRIBUTE → PRODUCT
    ########################################

    for attr in attributes:

        for prod in products:

            sim = safe_cosine(attr["embedding"], prod["embedding"])

            dist = distance_score(attr, prod)

            score = sim

            G.add_edge(
                attr["text"],
                prod["text"],
                relation="attribute_product",
                weight=score,
                sim=sim,
                dist_score=dist
            )

    ########################################
    # PRODUCT → PRODUCT
    ########################################

    for i in range(len(products)):

        for j in range(i + 1, len(products)):

            p1 = products[i]
            p2 = products[j]

            sim = safe_cosine(p1["embedding"], p2["embedding"])
            dist = distance_score(p1, p2)

            score = sim

            G.add_edge(
                p1["text"],
                p2["text"],
                relation="product_product",
                weight=score,
                sim=sim,
                dist_score=dist
            )

            G.add_edge(
                p2["text"],
                p1["text"],
                relation="product_product",
                weight=score,
                sim=sim,
                dist_score=dist
            )

    return G


############################################
# GET THRESHOLD
############################################

def get_threshold(label):

    return THRESHOLD_MAP.get(label, DEFAULT_THRESHOLD)


############################################
# LINK ENTITIES
############################################

def link_entities(text, entities):

    entities = generate_span_embeddings(text, entities)

    G = build_graph(entities)

    results = []

    node_labels = {e["text"]: e["label"] for e in entities}

    for node in G.nodes:

        edges = G.out_edges(node, data=True)

        if not edges:
            continue

        best = max(edges, key=lambda x: x[2]["weight"])

        label = node_labels.get(best[0], None)

        threshold = get_threshold(label)

        if best[2]["weight"] >= threshold:

            results.append({
                "attribute": best[0],
                "product": best[1],
                "score": round(best[2]["weight"], 3),
                "threshold": threshold
            })

    return results


############################################
# FINAL FORMATTER
############################################

def build_final_json(text, entities, linker_results):

    final = {
        "intent": "product_search",
        "entities": []
    }

    product_like = [
        e for e in entities
        if e["label"] in ["product_name", "accessory"]
    ]

    attributes = [
        e for e in entities
        if e["label"] not in ["product_name", "accessory"]
    ]

    ########################################
    # SKIP LINKER IF ONLY ONE PRODUCT
    ########################################

    if len(product_like) == 1:

        prod = product_like[0]

        product_entry = {
            "id": 1,
            "product_name": prod["text"],
            "is_main_product": prod.get("is_main_product", True)
        }

        for attr in attributes:
            product_entry[attr["label"]] = attr["text"]

        final["entities"].append(product_entry)

        return json.dumps(final, indent=2)

    ########################################
    # NORMAL CASE
    ########################################

    product_map = {}
    id_counter = 1

    for ent in entities:

        if ent["label"] == "product_name":

            obj = {
                "id": id_counter,
                "product_name": ent["text"],
                "is_main_product": ent.get("is_main_product", False)
            }

            product_map[ent["text"]] = obj

            final["entities"].append(obj)

            id_counter += 1

    for link in linker_results:

        attr = link["attribute"]
        prod = link["product"]

        attr_label = None

        for e in entities:
            if e["text"] == attr:
                attr_label = e["label"]

        if attr_label == "accessory":

            parent_id = product_map[prod]["id"]

            accessory_entry = {
                "id": id_counter,
                "product_name": attr,
                "is_main_product": False,
                "is_related_to": parent_id
            }

            final["entities"].append(accessory_entry)

            id_counter += 1

        elif attr_label not in ["product_name"]:

            product_map[prod][attr_label] = attr

    return json.dumps(final, indent=2)


############################################
# PIPELINE
############################################

def run_pipeline(text, entities):

    product_like = [
        e for e in entities
        if e["label"] in ["product_name", "accessory"]
    ]

    if len(product_like) == 1:

        return build_final_json(text, entities, [])

    linker_results = link_entities(text, entities)

    return build_final_json(text, entities, linker_results)


############################################
# TEST
############################################

text = "8gb samsung galaxy s23 vs 32gb chromebook"

entities = [

 {'label': 'memory', 'text': '8gb', 'start': 0, 'end': 3},

 {'label': 'product_name',
  'text': 'samsung galaxy s23',
  'start': 4,
  'end': 22,
  "is_main_product": True},

 {'label': 'memory', 'text': '32gb', 'start': 26, 'end': 30},

 {'label': 'product_name',
  'text': 'chromebook',
  'start': 31,
  'end': 41,
  "is_main_product": False}

]

start = time.perf_counter()

output = run_pipeline(text, entities)

end = time.perf_counter()

print(output)

print("\nInference Time:", (end-start)*1000, "ms")
