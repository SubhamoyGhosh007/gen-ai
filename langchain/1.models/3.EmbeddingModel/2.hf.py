from langchain_huggingface import  HuggingFaceEndpointEmbeddings 
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpointEmbeddings (
    model = "sentence-transformers/all-MiniLM-L6-v2"
)

print(str(llm.embed_query("I am Subhamoy Ghosh.")))