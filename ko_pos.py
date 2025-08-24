from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch, json

# ✅ lightweight open-source Korean instruction-tuned LLM
MODEL_NAME = "beomi/KoAlpaca-Polyglot-1.3B"

# 🔄 Switch here: True = force GPU, False = CPU
USE_GPU = True

# -------------------------------
# Device selection
# -------------------------------
if USE_GPU and torch.cuda.is_available():
    device = "cuda"
    device_index = 0
else:
    device = "cpu"
    device_index = -1

print(f"🚀 Loading {MODEL_NAME} on {device.upper()} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if device == "cuda":
    model = model.to("cuda")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device_index
)

# -------------------------------
# Function to extract modifiers
# -------------------------------
def extract_modifiers_llm(sentence: str):
    prompt = f"""
    문장에서 명사를 찾아서 JSON 형식으로 출력하세요.
    각 명사에 대해:
    - 관련 형용사(수식어)
    - 관련 조사 (조사/격조사)
    - 루트 명사 여부 (is_root: true/false)

    문장: "{sentence}"
    
    출력 형식:
    [
      {{
        "noun": "...",
        "modifiers": ["..."],
        "adpositions": ["..."],
        "is_root": true/false
      }}
    ]
    """

    output = pipe(prompt, max_new_tokens=256, do_sample=False)[0]['generated_text']

    try:
        json_str = output[output.index("[") : output.rindex("]")+1]
        return json.loads(json_str)
    except Exception:
        return {"raw_output": output}


# -------------------------------
# ✅ Example Run
# -------------------------------
sentences = [
    "저렴한 냉장고 보여줘",
    "삼성 S24폰을 사고 싶어",
    "큰 집에서 살고 싶다"
]

for s in sentences:
    print(f"\nSentence: {s}")
    result = extract_modifiers_llm(s)
    print(json.dumps(result, ensure_ascii=False, indent=2))
