# Contextual compression retrieves broad results, then keeps only
# query-relevant passages from each result.
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.documents import Document
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

# Load GOOGLE_API_KEY from the local .env file.
load_dotenv()

# These mixed-topic documents demonstrate why compression is useful.
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

# Create a FAISS vector store from the documents.
embedding_model = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2-preview')
vectorstore = FAISS.from_documents(docs, embedding_model)

# Retrieve several broad candidates before compressing their contents.
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# The LLM extracts only text relevant to the user's question.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
compressor = LLMChainExtractor.from_llm(llm)

# Combine the base retriever and the document compressor.
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# Only the photosynthesis passages should remain in the compressed results.
query = "What is photosynthesis?"
compressed_results = compression_retriever.invoke(query)

# Display the shortened, query-focused documents.
for i, doc in enumerate(compressed_results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
