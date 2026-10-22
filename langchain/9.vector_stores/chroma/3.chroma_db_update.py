# Chroma provides persistence and document update operations.
from langchain_chroma import Chroma
# The embedding model is used by the collection for vector operations.
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# A replacement Document keeps the same metadata schema as the original.
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load GOOGLE_API_KEY and other local settings from .env.
load_dotenv()

# Reopen the collection containing the document to update.
vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model='gemini-embedding-2-preview'),
    persist_directory='my_chorma_db',
    collection_name='sample'
)

# Replace the text and metadata for the existing document ID.
updated_doc = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)

# The ID must already exist in the persistent collection.
vector_store.update_document(document_id='a4e2d660-55cd-4a27-93f8-f9a5bf0b43c9', document=updated_doc)

# get_by_ids expects a list, even when retrieving only one document.
updated_docs = vector_store.get_by_ids(
    ['a4e2d660-55cd-4a27-93f8-f9a5bf0b43c9']
)
# Confirm that the updated document is now stored.
print(updated_docs)