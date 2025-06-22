from ts.torch_handler.base_handler import BaseHandler
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import time

class TextClassificationHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        model_dir = ctx.system_properties.get("model_dir")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float,
        local_files_only=True,        # prevent internet calls
        from_safetensors=True         # explicit support for safetensors
        )
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        text = data[0].get("data") or data[0].get("body")
        text = text.decode("utf-8") if isinstance(text, bytes) else text

        # ✅ Remove all symbols and strip leading/trailing spaces
        cleaned = re.sub(r"[^\w\s]", "", text).strip()

        return self.tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)



    def inference(self, inputs):
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
        end_time = time.time()

        pred = torch.argmax(outputs.logits, dim=1).item()
        inference_time = round((end_time - start_time) * 1000, 2)  # ms

        return pred, inference_time

    def postprocess(self, inference_output):
        pred, inference_time = inference_output
        label = "NLQ" if pred == 1 else "KH"
        return {
            "prediction": label,
            "inference_time_ms": inference_time
        }
