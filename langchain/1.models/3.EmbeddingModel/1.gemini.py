# Embeddings map text to vectors for semantic search and similarity.
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load provider credentials from the local environment.
load_dotenv()

# Configure the Gemini embedding model.
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

# Embed one query string into a numeric vector.
response = embedding.embed_query("Delhi is the capital of India.")

print(str(response))