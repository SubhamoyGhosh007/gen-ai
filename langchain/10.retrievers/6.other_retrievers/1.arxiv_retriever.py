"""Retrieve research-paper summaries directly from the arXiv API."""

from langchain_community.retrievers import ArxivRetriever

# ArxivRetriever searches arXiv rather than a local vector store.
retriever = ArxivRetriever(
    top_k_results=2,
    load_max_docs=2,
    get_full_documents=False,
)

# The query is sent to arXiv's search service.
query = "retrieval augmented generation for large language models"
documents = retriever.invoke(query)

# Each result is a LangChain Document with paper metadata.
for index, document in enumerate(documents, start=1):
    print(f"\n--- arXiv result {index} ---")
    print(f"Title: {document.metadata.get('Title', 'Unknown')}")
    print(f"Authors: {document.metadata.get('Authors', 'Unknown')}")
    print(document.page_content)
