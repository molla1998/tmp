import os
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort

from transformers import AutoTokenizer, AutoModel

############################################################
# CONFIG
############################################################

MODEL_PATH = "saved_models/best_model.pt"
ONNX_PATH = "multitask_model.onnx"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MAIN_PRODUCT_LABELS = {
    "PRODUCT_NAME",
    "ACCESSORY"
}

############################################################
# LOAD CHECKPOINT
############################################################

print("Loading checkpoint...")

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

        self.ner_head = nn.Linear(
            hidden_size,
            len(label2id)
        )

        self.intent_head = nn.Linear(
            hidden_size,
            len(intent2id)
        )

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

        seq_output = outputs.last_hidden_state

        cls_output = seq_output[:, 0]

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

print("Loading pytorch model...")

model = MultiTaskModel()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)

model.eval()

############################################################
# EXPORT ONNX
############################################################

if not os.path.exists(ONNX_PATH):

    print("Exporting ONNX...")

    dummy_input_ids = torch.randint(
        0,
        1000,
        (1, CONFIG["max_length"]),
        dtype=torch.long
    ).to(DEVICE)

    dummy_attention_mask = torch.ones(
        (1, CONFIG["max_length"]),
        dtype=torch.long
    ).to(DEVICE)

    torch.onnx.export(
        model,
        (
            dummy_input_ids,
            dummy_attention_mask
        ),
        ONNX_PATH,
        input_names=[
            "input_ids",
            "attention_mask"
        ],
        output_names=[
            "ner_logits",
            "intent_logits",
            "main_logits"
        ],
        dynamic_axes={
            "input_ids": {
                0: "batch",
                1: "seq_len"
            },
            "attention_mask": {
                0: "batch",
                1: "seq_len"
            },
            "ner_logits": {
                0: "batch",
                1: "seq_len"
            },
            "main_logits": {
                0: "batch",
                1: "seq_len"
            },
            "intent_logits": {
                0: "batch"
            }
        },
        opset_version=14
    )

    print("ONNX Export Complete")

############################################################
# LOAD ONNX
############################################################

print("Loading ONNX Runtime...")

session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CPUExecutionProvider"]
)

############################################################
# POST PROCESS
############################################################

def post_process(
    text,
    offsets,
    ner_tags,
    main_preds
):

    entities = []

    current = None

    for tag, offset, main_pred in zip(
        ner_tags,
        offsets,
        main_preds
    ):

        start, end = offset

        if start == end:
            continue

        if tag == "O":

            if current:
                entities.append(current)
                current = None

            continue

        prefix, label = tag.split("-", 1)

        ####################################################
        # BEGIN
        ####################################################
        if prefix == "B":

            if current:
                entities.append(current)

            current = {
                "entity": label,
                "start": int(start),
                "end": int(end),
                "text": text[start:end]
            }

            if label in MAIN_PRODUCT_LABELS:

                current[
                    "is_main_product"
                ] = bool(main_pred)

        ####################################################
        # INSIDE
        ####################################################
        elif prefix == "I" and current:

            current["end"] = int(end)

            current["text"] = text[
                current["start"]:end
            ]

            if label in MAIN_PRODUCT_LABELS:

                current[
                    "is_main_product"
                ] = (
                    current.get(
                        "is_main_product",
                        False
                    )
                    or bool(main_pred)
                )

    if current:
        entities.append(current)

    return entities

############################################################
# ONNX PREDICT
############################################################

def predict(query):

    encoding = tokenizer(
        query,
        truncation=True,
        padding="max_length",
        max_length=CONFIG["max_length"],
        return_offsets_mapping=True
    )

    input_ids = np.array(
        [encoding["input_ids"]],
        dtype=np.int64
    )

    attention_mask = np.array(
        [encoding["attention_mask"]],
        dtype=np.int64
    )

    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    )

    ner_logits = outputs[0]
    intent_logits = outputs[1]
    main_logits = outputs[2]

    ########################################################
    # ARGMAX
    ########################################################

    ner_preds = np.argmax(
        ner_logits,
        axis=-1
    )[0]

    intent_pred = np.argmax(
        intent_logits,
        axis=-1
    )[0]

    main_preds = np.argmax(
        main_logits,
        axis=-1
    )[0]

    ########################################################
    # LABELS
    ########################################################

    ner_tags = [
        id2label[i]
        for i in ner_preds
    ]

    entities = post_process(
        text=query,
        offsets=encoding["offset_mapping"],
        ner_tags=ner_tags,
        main_preds=main_preds
    )

    return {
        "intent": id2intent[intent_pred],
        "entities": entities
    }

############################################################
# TEST
############################################################

while True:

    query = input("\nQuery: ")

    if query.lower() == "exit":
        break

    result = predict(query)

    print("\nRESULT")
    print(result)
