import json
import numpy as np
import onnxruntime as ort

from transformers import AutoTokenizer


class ProductNLU:

    MAIN_PRODUCT_LABELS = {
        "PRODUCT_NAME",
        "ACCESSORY"
    }

    def __init__(
        self,
        model_path="deployment/model.onnx",
        tokenizer_path="deployment/tokenizer",
        labels_path="deployment/labels.json",
        config_path="deployment/config.json"
    ):

        # -------------------------
        # Config
        # -------------------------
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # -------------------------
        # Labels
        # -------------------------
        with open(labels_path, "r") as f:
            labels = json.load(f)

        self.id2label = {
            int(k): v
            for k, v in labels["id2label"].items()
        }

        self.id2intent = {
            int(k): v
            for k, v in labels["id2intent"].items()
        }

        # -------------------------
        # Tokenizer
        # -------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path
        )

        # -------------------------
        # ONNX Runtime
        # -------------------------
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

    # ====================================================
    # POST PROCESS
    # ====================================================

    def _decode_entities(
        self,
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

            if "-" not in tag:
                continue

            prefix, label = tag.split("-", 1)

            ################################################
            # BEGIN
            ################################################
            if prefix == "B":

                if current:
                    entities.append(current)

                current = {
                    "entity": label,
                    "value": text[start:end],
                    "start": int(start),
                    "end": int(end)
                }

                if label in self.MAIN_PRODUCT_LABELS:
                    current["is_main_product"] = bool(
                        main_pred
                    )

            ################################################
            # INSIDE
            ################################################
            elif prefix == "I" and current:

                current["end"] = int(end)

                current["value"] = text[
                    current["start"]:end
                ]

                if (
                    label
                    in self.MAIN_PRODUCT_LABELS
                ):
                    current["is_main_product"] = (
                        current.get(
                            "is_main_product",
                            False
                        )
                        or bool(main_pred)
                    )

        if current:
            entities.append(current)

        return entities

    # ====================================================
    # PREDICT
    # ====================================================

    def predict(self, query: str):

        encoding = self.tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=self.config[
                "max_length"
            ],
            return_offsets_mapping=True
        )

        input_ids = np.asarray(
            [encoding["input_ids"]],
            dtype=np.int64
        )

        attention_mask = np.asarray(
            [encoding["attention_mask"]],
            dtype=np.int64
        )

        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        )

        ner_logits = outputs[0]
        intent_logits = outputs[1]
        main_logits = outputs[2]

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

        ner_tags = [
            self.id2label[idx]
            for idx in ner_preds
        ]

        entities = self._decode_entities(
            text=query,
            offsets=encoding[
                "offset_mapping"
            ],
            ner_tags=ner_tags,
            main_preds=main_preds
        )

        return {
            "query": query,
            "intent": self.id2intent[
                intent_pred
            ],
            "entities": entities
        }
