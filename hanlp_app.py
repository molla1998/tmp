import streamlit as st
import hanlp
import torch
from pathlib import Path


############################################
# CONFIG
############################################
# CHANGE THIS to your local extracted folder
# OR leave None to auto download
LOCAL_MODEL_PATH = None
# e.g. LOCAL_MODEL_PATH = r"D:\models\hanlp\ud_ontonotes_tok_pos_lem_fea_ner_srl_dep_sdp_con_xlmr_base"


############################################
# LOAD MODEL
############################################
@st.cache_resource
def load_model(use_gpu: bool):
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if LOCAL_MODEL_PATH and Path(LOCAL_MODEL_PATH).exists():
        model = hanlp.load(LOCAL_MODEL_PATH)
    else:
        model = hanlp.load(
            hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE
        )
    model.to(device)
    return model, str(device)


model = None


############################################
# NLP LOGIC
############################################
def analyze(sentence, nouns, model):
    result = model([sentence])

    tokens = result["tok"][0]
    pos = result["pos"][0]
    deps = result["dep"][0]  # list of (head, rel)

    # Build index map
    noun_results = []

    for noun in nouns:
        noun = noun.strip()
        matches = [i for i, t in enumerate(tokens) if t.lower() == noun.lower()]
        if not matches:
            noun_results.append({
                "noun": noun,
                "found": False,
                "adjectives": [],
                "adpositions": [],
                "root_related": False
            })
            continue

        idx = matches[0]  # first occurrence
        head_index, rel = deps[idx]

        # collect children deps
        children = [
            (i, d) for i, d in enumerate(deps)
            if d[0] == idx + 1  # dep heads are 1-based
        ]

        adjectives = []
        adpositions = []

        for child_idx, (h, r) in children:
            if r.startswith("amod") or pos[child_idx].startswith("ADJ"):
                adjectives.append(tokens[child_idx])
            if r in ["case", "adp", "mark"]:
                adpositions.append(tokens[child_idx])

        # Is related to root?
        root_related = (head_index == 0) or any(
            dep[0] == 0 for dep in deps
        )

        noun_results.append({
            "noun": noun,
            "found": True,
            "adjectives": adjectives,
            "adpositions": adpositions,
            "root_related": root_related
        })

    return tokens, pos, deps, noun_results


############################################
# STREAMLIT UI
############################################
st.set_page_config(page_title="HanLP KO+EN Dependency App", layout="centered")
st.title("HanLP POS + Dependency Explorer")
st.write("Supports **Korean + English mixed sentences**")

use_gpu = st.toggle("Use GPU (if available)", value=False)

if st.button("Initialize / Reload Model"):
    model, device_name = load_model(use_gpu)
    st.success(f"Model loaded on: {device_name}")

if "model" not in globals() or model is None:
    model, device_name = load_model(use_gpu)

sentence = st.text_area("Enter mixed English + Korean sentence:",
                        value="I will 학교에 갈 거예요 tomorrow because the weather is 좋다.",
                        height=120)

noun_input = st.text_input(
    "Comma-separated noun list (match tokens):",
    value="weather, 학교, tomorrow"
)

if st.button("Analyze"):
    nouns = [n.strip() for n in noun_input.split(",") if n.strip()]
    tokens, pos, deps, results = analyze(sentence, nouns, model)

    st.subheader("Tokens")
    st.write(tokens)

    st.subheader("POS")
    st.write(pos)

    st.subheader("Dependencies (head_index, relation)")
    st.write(deps)

    st.subheader("Results Per Noun")
    for r in results:
        st.markdown(f"### `{r['noun']}`")
        if not r["found"]:
            st.error("Not found in tokens")
            continue

        st.write("**Adjectives:**", r["adjectives"] or "None")
        st.write("**Adpositions / Case:**", r["adpositions"] or "None")
        st.write("**Related to Root?:**", "✅ Yes" if r["root_related"] else "❌ No")
