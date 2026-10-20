# Adapt a hosted Hugging Face text model to LangChain's chat interface.
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load the Hugging Face authentication token from the environment.
load_dotenv()

# Configure the hosted endpoint and its generation task.
llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

# Wrap the endpoint so it behaves like a LangChain chat model.
model = ChatHuggingFace(llm = llm)

# Invoke the chat model and print its text content.
response = model.invoke("What is the capital of India?")
print(response.content)