from typing import List
from pydantic import BaseModel, Field
from transformers import pipeline  # For your local BERT NSFW classifier
from langchain_openai import ChatOpenAI  # For vLLM (OpenAI-compatible server format)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

# ==========================================
# 1. DEFINE YOUR EXACT DESIRED OUTPUT SCHEMA
# ==========================================
class NEREntity(BaseModel):
    text: str = Field(description="The substring from the query extracted as an entity.")
    label: str = Field(description="The entity category (e.g., PERSON, ORG, LOC, PRODUCT, DATE).")

class AnalysisOutput(BaseModel):
    """The master JSON schema required downstream by your app interface."""
    language: str = Field(description="The detected language of the query.")
    intent: str = Field(description="The parsed intent or goal of the user query.")
    search_tab: str = Field(description="The target UI domain filter tab: 'All', 'Images', 'News', 'Videos', or 'Maps'.")
    ner: List[NEREntity] = Field(default=[], description="List of Named Entities extracted from the query.")

# ==========================================
# 2. INITIALIZE YOUR LOCAL MODERN HARDWARE MODELS
# ==========================================

# A. Local BERT NSFW Classifier Initialization
# (Assumes standard transformers wrapper, or substitute with your exact local weights path)
print("Initializing local BERT NSFW Classification pipeline...")
bert_nsfw_classifier = pipeline("text-classification", model="unitary/toxic-bert")

def nsfw_guardrail_node(input_dict: dict) -> dict:
    """Invokes your local BERT model to tag incoming content safety status."""
    query = input_dict["user_query"]
    
    # Run prediction through local BERT weights
    prediction = bert_nsfw_classifier(query)[0]
    
    # Check if toxicity score crosses threshold or label matches unsafe
    # Adjust condition keys depending on your exact fine-tuned BERT labels
    is_unsafe = prediction["label"] == "toxic" and prediction["score"] > 0.7
    
    input_dict["is_nsfw"] = is_unsafe
    return input_dict


# B. Local vLLM Gemma 4 Server Connector
print("Connecting LangChain backbone to local vLLM Gemma 4 instance...")
# vLLM exposes an OpenAI-compatible path, letting us utilize standard high-throughput tooling
vllm_gemma4 = ChatOpenAI(
    model="google/gemma-4-26b-it", 
    openai_api_base="http://localhost:8000/v1", # The standard local vLLM route
    openai_api_key="local-vllm-token",          # Bypass key placeholder
    temperature=0.0
)

# Enforce structured constraints on your local Gemma 4 engine using Pydantic
structured_gemma = vllm_gemma4.with_structured_output(AnalysisOutput)

# ==========================================
# 3. PROMPTS & RUNNABLE SUB-CHAINS
# ==========================================

system_prompt = (
    "You are a specialized fine-tuned processing backbone. Inspect the user query and "
    "extract the primary language, the core user intent, the ideal Search Tab UI context, "
    "and any clear Named Entities (NER) present in the text."
)

gemma_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{user_query}")
])

# The standard safe-execution path
gemma_processing_chain = gemma_prompt | structured_gemma


# Safe fallback function if local BERT trips the alarm
def safety_block_fallback(input_dict: dict) -> AnalysisOutput:
    return AnalysisOutput(
        language="Unknown",
        intent="TERMINATED_BY_LOCAL_BERT_NSFW_FILTER",
        search_tab="None",
        ner=[]
    )

# ==========================================
# 4. UNIFIED LANGCHAIN PIPELINE ASSEMBLY (LCEL)
# ==========================================

# Conditional branch checking the state passed by the BERT node
routing_gate = RunnableBranch(
    (lambda state: state["is_nsfw"] == True, safety_block_fallback),
    gemma_processing_chain # Else, let Gemma 4 generate the structured insights
)

# Direct linear pipeline execution tree
query_understanding_pipeline = (
    {"user_query": RunnablePassthrough()}  # 1. Take raw input string
    | nsfw_guardrail_node                   # 2. Mutate state through BERT filter
    | routing_gate                         # 3. Choose branch based on safety outcomes
)

# ==========================================
# 5. TEST RUNS
# ==========================================
if __name__ == "__main__":
    # Test 1: Standard Search Intent
    clean_test = "Show me recent news about Gemma 4 performance metrics"
    print(f"\nEvaluating query: '{clean_test}'")
    output_clean = query_understanding_pipeline.invoke(clean_test)
    print(output_clean.model_dump_json(indent=2))

    print("\n" + "="*50 + "\n")

    # Test 2: Triggering Unsafe Input
    unsafe_test = "Go away you stupid idiot bot I hate you" # Simple toxicity trigger for default BERT
    print(f"Evaluating query: '{unsafe_test}'")
    output_unsafe = query_understanding_pipeline.invoke(unsafe_test)
    print(output_unsafe.model_dump_json(indent=2))
