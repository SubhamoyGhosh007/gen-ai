# Chroma manages the persistent vector collection.
from langchain_chroma import Chroma
# The same embedding model is configured when reopening the collection.
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load GOOGLE_API_KEY and other local settings from .env.
load_dotenv()

# Open the existing collection containing the document to delete.
vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model='gemini-embedding-2-preview'),
    persist_directory='my_chorma_db',
    collection_name='sample'
)

# IDs must be passed as a list; this permanently removes the document.
vector_store.delete(ids=['a4e2d660-55cd-4a27-93f8-f9a5bf0b43c9'])

# Inspect the remaining documents after deletion.
rows = vector_store.get(include=['embeddings','documents', 'metadatas'])
print(rows)