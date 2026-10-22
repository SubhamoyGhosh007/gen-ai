# FAISS stores vectors in memory and exposes a retriever interface.
from langchain_community.vectorstores import FAISS
# Embeddings convert documents and queries into comparable numeric vectors.
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Document holds text that can be indexed and returned by the retriever.
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load GOOGLE_API_KEY from the local .env file.
load_dotenv()

# These are the source documents that will be indexed.
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# Use the same embedding model for indexing and searching.
embedding_model = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2-preview')

# Build an in-memory FAISS index from the documents.
vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model,
)

# Wrap the vector store as a Runnable retriever that returns two results.
retriever = vector_store.as_retriever(search_kwargs={'k':2})
# The query is embedded and compared with the indexed document vectors.
query = "What is Chroma used for?"
results = retriever.invoke(query)

# Print the documents selected by semantic similarity.
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
    
# A retriever can be composed directly into LangChain chains.