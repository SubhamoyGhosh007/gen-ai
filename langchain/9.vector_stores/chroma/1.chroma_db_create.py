# Google embeddings convert each document into a vector for similarity search.
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Chroma stores vectors and the original documents on disk.
from langchain_chroma import Chroma
# Document stores searchable text together with metadata.
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load GOOGLE_API_KEY and other local settings from .env.
load_dotenv()

# Create documents with team metadata that can later be used for filtering.
doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs = [doc1, doc2, doc3, doc4, doc5]

# Reuse a persistent collection so data remains available between runs.
vector_stores = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model='gemini-embedding-2-preview'),
    persist_directory='my_chorma_db',
    collection_name='sample'
)

# Embed and store all documents; Chroma generates an ID for each one.
doc_ids = vector_stores.add_documents(docs)

# Keep these IDs for later update or delete operations.
print("Document IDs: ", doc_ids)
