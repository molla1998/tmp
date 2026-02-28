import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "relation_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128

# ==============================
# LOAD MODEL
# ==============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# ==============================
# LABEL MAP
# ==============================
label_map = {
    0: "No Relation",
    1: "P2 depends on P1",
    2: "P1 depends on P2"
}

# ==============================
# INFERENCE FUNCTION
# ==============================
def predict_relation(query, product_1, product_2):

    encoding = tokenizer(
        query,
        product_1,
        product_2,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return {
        "predicted_label": predicted_class,
        "relation": label_map[predicted_class],
        "confidence": round(confidence, 4)
    }


# ==============================
# EXAMPLE
# ==============================
if __name__ == "__main__":

    query = "show me s23 phone with charger"
    product_1 = "s23 phone"
    product_2 = "charger"

    result = predict_relation(query, product_1, product_2)

    print("\nPrediction Result:")
    print(result)
