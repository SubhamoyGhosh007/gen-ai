# Use a hosted Hugging Face model to create text embeddings.
from langchain_huggingface import  HuggingFaceEndpointEmbeddings 
from dotenv import load_dotenv

# Load the Hugging Face authentication token from the environment.
load_dotenv()

# Configure the sentence-transformer embedding endpoint.
llm = HuggingFaceEndpointEmbeddings (
    model = "sentence-transformers/all-MiniLM-L6-v2"
)

# Embed a single sentence for downstream vector operations.
print(str(llm.embed_query("I am Subhamoy Ghosh.")))