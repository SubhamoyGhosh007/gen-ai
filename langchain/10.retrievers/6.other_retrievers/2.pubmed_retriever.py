"""Retrieve biomedical article summaries directly from PubMed."""

from langchain_community.retrievers import PubMedRetriever

# PubMedRetriever queries the PubMed API and returns article Documents.
retriever = PubMedRetriever(
    top_k_results=2,
)

# PubMed searches biomedical and life-science literature.
query = "effects of regular exercise on cardiovascular health"
documents = retriever.invoke(query)

# Each result contains article text and metadata such as its PubMed ID.
for index, document in enumerate(documents, start=1):
    print(f"\n--- PubMed result {index} ---")
    print(f"PMID: {document.metadata.get('uid', 'Unknown')}")
    print(document.page_content)
