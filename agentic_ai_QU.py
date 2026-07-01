import json
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


# ==========================================================
# Output Schema
# ==========================================================

class ProductQuery(BaseModel):
    intent: str
    category: Optional[str] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    price_range: Optional[Dict[str, Any]] = None
    sort: Optional[str] = None


# ==========================================================
# System Prompt
# ==========================================================

SYSTEM_PROMPT = """
You are a Query Understanding Agent.

Your task is to convert a user's product search into structured JSON.

Rules:
- Return ONLY valid JSON.
- Do NOT explain your answer.
- Do NOT include markdown.
- Extract as much information as possible.

Output Schema:

{
    "intent": "product_search",
    "category": "",
    "brand": "",
    "model": "",
    "attributes": {},
    "price_range": null,
    "sort": null
}

Example

User:
show me samsung s24 phone with 8gb ram

Output:
{
    "intent":"product_search",
    "category":"phone",
    "brand":"Samsung",
    "model":"S24",
    "attributes":{
        "ram":"8GB"
    },
    "price_range":null,
    "sort":null
}
"""


# ==========================================================
# Prompt Template
# ==========================================================

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{query}"),
    ]
)


# ==========================================================
# Fake LLM
# ==========================================================

class FakeLLM:
    """
    Production-friendly mock.
    Replace this class with ChatOpenAI later.
    """

    def invoke(self, prompt_value):
        print("\n========== PROMPT SENT TO LLM ==========\n")
        print(prompt_value)
        print("\n========================================\n")

        return json.dumps(
            {
                "intent": "product_search",
                "category": "phone",
                "brand": "Samsung",
                "model": "S24",
                "attributes": {
                    "ram": "8GB"
                },
                "price_range": None,
                "sort": None,
            }
        )


# ==========================================================
# Agent
# ==========================================================

class QueryUnderstandingAgent:

    def __init__(self):
        self.llm = FakeLLM()

    def understand(self, query: str) -> ProductQuery:

        formatted_prompt = prompt.invoke(
            {
                "query": query
            }
        )

        llm_response = self.llm.invoke(formatted_prompt)

        response = json.loads(llm_response)

        return ProductQuery.model_validate(response)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    agent = QueryUnderstandingAgent()

    query = input("Enter Query: ")

    result = agent.understand(query)

    print("\n========== Structured JSON ==========\n")
    print(result.model_dump_json(indent=4))
