pip install vllm langchain langchain-openai


python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2b \
  --port 8000

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/gemma-2b",
    token="your_token"
)
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/gemma-2b",
    local_dir="/home/yourname/models/gemma-2b",
    local_dir_use_symlinks=False,
    token="hf_xxxxx"   # optional if already logged in
)
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",  # required but ignored
    model="google/gemma-2b",
    temperature=0.7
)

response = llm.invoke("Explain what LangChain is in 2 lines")
print(response.content)

python -m vllm.entrypoints.openai.api_server --model google/gemma-2b --port 8000

set CUDA_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server --model /path/to/gemma-2b --port 8000
