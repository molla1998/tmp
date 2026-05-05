pip install vllm langchain langchain-openai


python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2b \
  --port 8000python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2b \
  --port 8000

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",  # required but ignored
    model="google/gemma-2b",
    temperature=0.7
)

response = llm.invoke("Explain what LangChain is in 2 lines")
print(response.content)
