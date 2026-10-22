# requests is used to recognize malformed responses from the Wikipedia API.
import requests
# wikipedia configures the API client's user agent and rate limiting.
import wikipedia
# This retriever searches Wikipedia directly; it does not use embeddings.
from langchain_community.retrievers import WikipediaRetriever

# Identify this application to Wikimedia when making API requests.
wikipedia.set_user_agent("gen-ai-tutorial/0.1")

# Pause between API calls to reduce the chance of Wikimedia rate limiting.
wikipedia.set_rate_limiting(True)

# Return at most two English Wikipedia articles for each query.
retriever = WikipediaRetriever(top_k_results=2, lang='en')

# Wikipedia interprets this as a natural-language search query.
query = "the geopolitical history of india and pakistan from the perspective of a chinese"

# The public API can return a non-JSON rate-limit response.
try:
    docs = retriever.invoke(query)
except requests.exceptions.JSONDecodeError as error:
    raise RuntimeError(
        "Wikipedia returned a non-JSON response, likely because its API "
        "rate limit was reached. Wait a little while and run this example again."
    ) from error

# Each result is a LangChain Document containing article text and metadata.
for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")