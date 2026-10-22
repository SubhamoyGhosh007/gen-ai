# MultiQueryRetriever asks an LLM to generate alternative query phrasings.
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

# Load GOOGLE_API_KEY from the local .env file.
load_dotenv()

# The extra phrasings help retrieve documents that use different wording.
all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Embed the local documents before creating the base retriever.
embedding_model = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2-preview')

# FAISS provides the underlying similarity search for each generated query.
vector_store = FAISS.from_documents(
    documents=all_docs,
    embedding=embedding_model,
)

# The LLM generates multiple queries; their results are merged and deduplicated.
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_type= 'mmr' ,search_kwargs={"k": 5, 'lambda_mult': 0.5}),
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
)

# This question can be expressed through several related concepts.
query = "How to improve energy levels and maintain balance?"
multiquery_results = multiquery_retriever.invoke(query)

# Print the unique documents returned across the generated queries.
for i, doc in enumerate(multiquery_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)