# app.py

import streamlit as st
import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModel
)

############################################################
# CONFIG
############################################################
MODEL_PATH = "saved_models/best_model.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

############################################################
# LOAD CHECKPOINT
############################################################
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

CONFIG = checkpoint["config"]

label2id = checkpoint["label2id"]
id2label = checkpoint["id2label"]

intent2id = checkpoint["intent2id"]
id2intent = checkpoint["id2intent"]

############################################################
# MAIN PRODUCT LABELS
############################################################
MAIN_PRODUCT_LABELS = {
    "PRODUCT_NAME",
    "ACCESSORY"
}

############################################################
# TOKENIZER
############################################################
tokenizer = AutoTokenizer.from_pretrained(
    CONFIG["model_name"]
)

############################################################
# MODEL
############################################################
class MultiTaskModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            CONFIG["model_name"]
        )

        hidden_size = (
            self.encoder.config.hidden_size
        )

        self.dropout = nn.Dropout(0.1)

        ####################################################
        # NER HEAD
        ####################################################
        self.ner_head = nn.Linear(
            hidden_size,
            len(label2id)
        )

        ####################################################
        # INTENT HEAD
        ####################################################
        self.intent_head = nn.Linear(
            hidden_size,
            len(intent2id)
        )

        ####################################################
        # MAIN PRODUCT HEAD
        ####################################################
        self.main_head = nn.Linear(
            hidden_size,
            2
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        seq_output = self.dropout(
            outputs.last_hidden_state
        )

        cls_output = self.dropout(
            seq_output[:, 0]
        )

        ner_logits = self.ner_head(
            seq_output
        )

        intent_logits = self.intent_head(
            cls_output
        )

        main_logits = self.main_head(
            seq_output
        )

        return (
            ner_logits,
            intent_logits,
            main_logits
        )

############################################################
# LOAD MODEL
############################################################
model = MultiTaskModel().to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

############################################################
# POST PROCESS
############################################################
def post_process(sample):

    entities = []

    current = None

    tokens = sample["tokens"]

    for tok in tokens:

        ner = tok["ner"]

        ####################################################
        # OUTSIDE
        ####################################################
        if ner == "O":

            if current:
                entities.append(current)
                current = None

            continue

        ####################################################
        # SPLIT BIO
        ####################################################
        prefix, label = ner.split("-", 1)

        ####################################################
        # BEGIN
        ####################################################
        if prefix == "B":

            if current:
                entities.append(current)

            current = {
                "entity": label,
                "text": tok["token"]
            }

            ################################################
            # main_product only for required labels
            ################################################
            if (
                label in MAIN_PRODUCT_LABELS
                and "is_main_product" in tok
            ):

                current["is_main_product"] = (
                    tok["is_main_product"]
                )

        ####################################################
        # INSIDE
        ####################################################
        elif prefix == "I" and current:

            token_text = tok["token"]

            ################################################
            # HANDLE WORDPIECE TOKENS
            ################################################
            if token_text.startswith("##"):

                current["text"] += (
                    token_text[2:]
                )

            else:

                current["text"] += (
                    " " + token_text
                )

            ################################################
            # merge main product
            ################################################
            if (
                label in MAIN_PRODUCT_LABELS
                and "is_main_product" in tok
            ):

                current["is_main_product"] = (
                    current.get(
                        "is_main_product",
                        False
                    )
                    or tok["is_main_product"]
                )

    ########################################################
    # LAST ENTITY
    ########################################################
    if current:
        entities.append(current)

    return {
        "intent": sample["intent"],
        "entities": entities
    }

############################################################
# PREDICT
############################################################
def predict(text):

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=CONFIG["max_length"]
    )

    input_ids = encoding[
        "input_ids"
    ].to(DEVICE)

    attention_mask = encoding[
        "attention_mask"
    ].to(DEVICE)

    ########################################################
    # FORWARD
    ########################################################
    with torch.no_grad():

        ner_logits, intent_logits, main_logits = model(
            input_ids,
            attention_mask
        )

    ########################################################
    # PREDICTIONS
    ########################################################
    ner_preds = torch.argmax(
        ner_logits,
        dim=-1
    )[0].cpu().numpy()

    intent_pred = torch.argmax(
        intent_logits,
        dim=-1
    ).item()

    main_preds = torch.argmax(
        main_logits,
        dim=-1
    )[0].cpu().numpy()

    ########################################################
    # TOKENS
    ########################################################
    tokens = tokenizer.convert_ids_to_tokens(
        input_ids[0]
    )

    results = []

    for token, ner_id, main_id in zip(
        tokens,
        ner_preds,
        main_preds
    ):

        item = {
            "token": token,
            "ner": id2label[ner_id]
        }

        ####################################################
        # main_product only for relevant labels
        ####################################################
        label = id2label[ner_id]

        if label.startswith("B-") or label.startswith("I-"):

            actual_label = label.split("-", 1)[1]

            if actual_label in MAIN_PRODUCT_LABELS:

                item["is_main_product"] = bool(
                    main_id
                )

        results.append(item)

    sample = {
        "intent": id2intent[intent_pred],
        "tokens": results
    }

    return post_process(sample)

############################################################
# STREAMLIT UI
############################################################
st.set_page_config(
    page_title="NER + Intent Model",
    layout="centered"
)

st.title("🧠 MultiTask NLP Model")

query = st.text_input(
    "Enter Query"
)

if st.button("Predict"):

    if query.strip():

        output = predict(query)

        st.subheader("Intent")
        st.json(output["intent"])

        st.subheader("Entities")
        st.json(output["entities"])

    else:

        st.warning(
            "Please enter a query"
        )
