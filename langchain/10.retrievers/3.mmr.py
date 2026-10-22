# Maximal Marginal Relevance (MMR) balances relevance with result diversity.
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load GOOGLE_API_KEY from the local .env file.
load_dotenv()

# Include related and unrelated documents to make MMR's behavior visible.
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

# Use Google embeddings to compare the query with each document.
embedding_model = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2-preview')

# Create the searchable FAISS index.
vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model,
)

# MMR selects relevant results while reducing near-duplicate content.
retriever = vector_store.as_retriever(
    search_type= 'mmr',
    search_kwargs = {
        'k':3,
        'lambda_mult': 0.5
    }  # k = result count; lambda_mult balances relevance and diversity
)

# Ask for documents about LangChain.
query = "What is langchain?"
results = retriever.invoke(query)

# Display the diverse results selected by MMR.
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)