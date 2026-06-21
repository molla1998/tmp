from typing import List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. DEFINE STRUTURED OUTPUT SCHEMAS
# ==========================================

class NEREntity(BaseModel):
    text: str = Field(description="The exact text snippet extracted as an entity.")
    label: str = Field(description="The type of entity (e.g., PERSON, ORG, LOC, PRODUCT, DATE).")

class AnalysisOutput(BaseModel):
    """The final structured JSON output schema requested."""
    language: str = Field(description="The detected language of the query.")
    intent: str = Field(description="The core intent or user goal behind the query.")
    search_tab: str = Field(description="The best UI category search tab for this query: 'All', 'Images', 'News', 'Videos', or 'Maps'.")
    ner: List[NEREntity] = Field(default=[], description="List of Named Entities extracted from the query.")

# ==========================================
# 2. DEFINE THE SHARED GRAPH STATE
# ==========================================

class PipelineState(TypedDict):
    user_query: str
    detected_language: Optional[str]
    is_nsfw: Optional[bool]
    nsfw_reason: Optional[str]
    final_json: Optional[AnalysisOutput]

# ==========================================
# 3. INTERACTION NODES (COMPONENTS)
# ==========================================

def language_detection_node(state: PipelineState):
    """Node 1: Fast deterministic language check or small model proxy."""
    query = state["user_query"]
    
    # In a real pipeline, you could use fasttext or a lightweight model here
    # For demonstration, we'll label it for routing, or let Gemma process it
    print(f"--- [LAYER 1] Detecting Language for: '{query}' ---")
    return {"detected_language": "Detected (Processing downstream)"}


def nsfw_filter_node(state: PipelineState):
    """Node 2: Guardrail filter to intercept safety violations immediately."""
    query = state["user_query"].lower()
    print("--- [LAYER 2] Running NSFW Policy Filters ---")
    
    # Example heuristic guardrail (Replace with ShieldGemma block in production)
    blacklisted_terms = ["darkweb market", "how to build a bomb", "exploit download"]
    
    if any(term in query for term in blacklisted_terms):
        return {
            "is_nsfw": True, 
            "nsfw_reason": "Query violates safety policy directives."
        }
    return {"is_nsfw": False}


def routing_decision_edge(state: PipelineState):
    """Conditional Edge: Diverts traffic instantly if content is marked unsafe."""
    if state["is_nsfw"]:
        return "blocked"
    return "continue"


def gemma_lora_inference_node(state: PipelineState):
    """Node 3: Primary task executor utilizing fine-tuned Gemma-3."""
    print("--- [LAYER 3] Executing Gemma-3 + LoRA Adapter Matrix Pass ---")
    query = state["user_query"]
    
    # Initialize your local model or API Client wrapper
    # Here, we utilize Gemma-3's structured output mechanism (.with_structured_output)
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Initialize Gemma-3 (or your local Hugging Face vLLM/Ollama target endpoint)
    llm = ChatGoogleGenerativeAI(model="gemma3-27b-it", temperature=0.0)
    structured_gemma = llm.with_structured_output(AnalysisOutput)
    
    system_prompt = (
        "You are a specialized fine-tuned processing backbone. Inspect the user query and "
        "extract the primary language, the core user intent, the ideal Search Tab UI context, "
        "and any clear Named Entities (NER) present in the text."
    )
    
    # Fire the optimized inference engine
    structured_response = structured_gemma.invoke([
        ("system", system_prompt),
        ("human", query)
    ])
    
    return {"final_json": structured_response}


def fallback_blocked_node(state: PipelineState):
    """Graceful error node when the safety filter trips."""
    print("!!! [ALERT] Pipeline terminated by Guardrails !!!")
    safe_fallback = AnalysisOutput(
        language="Unknown",
        intent="BLOCKED_BY_SAFETY_GUARDRAILS",
        search_tab="None",
        ner=[]
    )
    return {"final_json": safe_fallback}

# ==========================================
# 4. COMPILING THE STATE GRAPH
# ==========================================

builder = StateGraph(PipelineState)

# 1. Add all functional nodes
builder.add_node("detect_language", language_detection_node)
builder.add_node("nsfw_filter", nsfw_filter_node)
builder.add_node("gemma_inference", gemma_lora_inference_node)
builder.add_node("safety_block", fallback_blocked_node)

# 2. Establish linear flow entries
builder.add_edge(START, "detect_language")
builder.add_edge("detect_language", "nsfw_filter")

# 3. Inject conditional gate right after NSFW filter step
builder.add_conditional_edges(
    "nsfw_filter",
    routing_decision_edge,
    {
        "blocked": "safety_block",
        "continue": "gemma_inference"
    }
)

# 4. Seal execution flow into terminate endpoints
builder.add_edge("gemma_inference", END)
builder.add_edge("safety_block", END)

# Compile into an actionable runtime engine
compiled_pipeline = builder.compile()

# ==========================================
# 5. EXECUTION & VERIFICATION
# ==========================================
if __name__ == "__main__":
    # Test Pass Case
    sample_input = {"user_query": "Did Sundar Pichai announce Gemma 4 in Mountain View last week?"}
    
    print("REQUISITIONING PASS 1:")
    result = compiled_pipeline.invoke(sample_input)
    
    print("\n--- STALWART STRUCTURED OUTPUT RESULT ---")
    # This automatically renders as the clean JSON object structured down below
    print(result["final_json"].model_dump_json(indent=2))