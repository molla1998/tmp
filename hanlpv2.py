import hanlp
import json
import torch

# =====================================================
# FORCE CPU ONLY
# =====================================================
device = torch.device("cpu")
torch.set_num_threads(4)

model = hanlp.load(
    hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE
)
model.to(device)


# =====================================================
# KOREAN NORMALIZATION
# =====================================================
KOREAN_PARTICLES = (
    "이", "가", "을", "를", "은", "는",
    "의", "에", "에서", "으로", "로",
    "와", "과", "이다", "이야", "예요", "이에요"
)

def normalize_korean_noun(token: str):
    for p in KOREAN_PARTICLES:
        if token.endswith(p) and len(token) > len(p):
            return token[:-len(p)]
    return token


def soft_match(token: str, target: str):
    return token == target or token.startswith(target)


# =====================================================
# SECONDARY DEPENDENCY COLLECTOR (2-HOP)
# =====================================================
def collect_secondary_modifiers(
    target_idx,
    tokens,
    pos,
    deps,
    modifier_labels
):
    adjectives = []

    # Intermediate nouns modifying target
    intermediate_idxs = [
        i for i, (h, r) in enumerate(deps)
        if h == target_idx + 1 and r in ("nmod", "compound", "obl")
    ]

    # Adjectives modifying intermediate nouns
    for mid_idx in intermediate_idxs:
        for j, (h, r) in enumerate(deps):
            if h == mid_idx + 1:
                if r in modifier_labels or pos[j] in ("ADJ", "VERB"):
                    adjectives.append(tokens[j])

    return adjectives


# =====================================================
# MAIN EXTRACTION FUNCTION
# =====================================================
def extract_product_phrases_info(query: str, target_phrases_list):
    parsed = model([query])

    tokens = parsed["tok"][0]
    pos = parsed["pos"][0]
    deps = parsed["dep"][0]   # (head, relation) | head is 1-based, 0 = ROOT

    modifier_labels = {
        "amod", "nmod", "compound", "det", "acl", "advmod"
    }

    results = {}

    for target in target_phrases_list:
        adjectives_all = []
        adps_all = []
        is_main_product = False

        for i, token in enumerate(tokens):
            norm_token = normalize_korean_noun(token)

            if soft_match(norm_token, target):
                head_idx, dep_rel = deps[i]

                # -----------------------------
                # DIRECT CHILD MODIFIERS
                # -----------------------------
                for j, (h, r) in enumerate(deps):
                    if h == i + 1:
                        if r in modifier_labels or pos[j] in ("ADJ", "VERB"):
                            adjectives_all.append(tokens[j])

                        if r == "case" or pos[j] == "ADP":
                            adps_all.append(tokens[j])

                # -----------------------------
                # LEFT MODIFIER FALLBACK (KO)
                # -----------------------------
                if i > 0 and pos[i - 1] in ("ADJ", "VERB"):
                    adjectives_all.append(tokens[i - 1])

                # -----------------------------
                # SECONDARY (2-HOP) MODIFIERS
                # -----------------------------
                secondary_adjs = collect_secondary_modifiers(
                    i, tokens, pos, deps, modifier_labels
                )
                adjectives_all.extend(secondary_adjs)

                # -----------------------------
                # MAIN PRODUCT / ROOT CHECK
                # -----------------------------
                if (
                    dep_rel == "root"
                    or head_idx == 0
                    or dep_rel in ("nsubj", "obj", "obl", "dep")
                    or (head_idx > 0 and deps[head_idx - 1][0] == 0)
                ):
                    is_main_product = True

        results[target] = {
            "adjectives": sorted(set(adjectives_all)),
            "adp": sorted(set(adps_all)),
            "is_main_product": is_main_product
        }

    return results


# =====================================================
# EXAMPLE USAGE
# =====================================================
if __name__ == "__main__":
    query = "품질이 좋은 태뷸리이 정말 저렴한 폰 추천해줘"
    targets = ["태뷸리", "폰"]

    results = extract_product_phrases_info(query, targets)

    print("Query:", query)
    print("Targets:", targets)
    print(json.dumps(results, ensure_ascii=False, indent=4))
