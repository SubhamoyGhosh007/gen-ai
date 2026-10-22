# Chroma is the vector store used for loading and searching saved documents.
from langchain_chroma import Chroma
# This embedding model must match the one used when the documents were added.
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load GOOGLE_API_KEY and other local settings from .env.
load_dotenv()

# Open the existing persistent collection instead of creating a new one.
vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model='gemini-embedding-2-preview'),
    persist_directory='my_chorma_db',
    collection_name='sample'
)

# Retrieve stored embeddings, text, and metadata for inspection.
rows = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
print(rows)

# Return the two documents most similar to the question.
row = vector_store.similarity_search(
    query='Who among these are bowlers ?',
    k=2
)
print(row)

# This variant also returns a distance score; lower values usually mean
# the result is more similar to the query.
row = vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)
print(row)

# Combine semantic search with a metadata filter for one team.
row = vector_store.similarity_search_with_score(
    query="Which Chennai Super Kings player is an all-rounder?",
    k=2,
    filter={"team": "Chennai Super Kings"},
)
print(row)