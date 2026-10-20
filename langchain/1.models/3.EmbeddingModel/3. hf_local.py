# Generate embeddings locally with a Hugging Face sentence-transformer.
from langchain_huggingface import HuggingFaceEmbeddings

# Loading the model locally avoids a remote inference request.
embedding = HuggingFaceEmbeddings(model_name= 'sentence-transformers/all-MiniLM-L6-v2')

# Multiple documents can be embedded in one batch.
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# Each document becomes a vector suitable for similarity search.
vector = embedding.embed_documents(documents)

print(str(vector))