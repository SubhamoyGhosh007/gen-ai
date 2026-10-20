from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import os

os.environ['HF_HOME'] = "/workspaces/gen-ai/hf_models"

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen3-0.6B",
    task= "text-generation",
    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens = 50
    )
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What is the capital of India?")
print(response.content)