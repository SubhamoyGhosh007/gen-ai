# Run a local Hugging Face pipeline through LangChain's chat adapter.
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import os

# Keep downloaded model files in a workspace-local cache.
os.environ['HF_HOME'] = "/workspaces/gen-ai/hf_models"

# Build a local text-generation pipeline with generation options.
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-0.6B",
    task= "text-generation",
    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens = 50
    )
)

# Expose the local pipeline through the common chat-model API.
model = ChatHuggingFace(llm=llm)

# Invoke the local model and display the generated message text.
response = model.invoke("What is the capital of India?")
print(response.content)