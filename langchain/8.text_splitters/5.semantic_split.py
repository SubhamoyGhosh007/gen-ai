# SemanticChunker compares the meaning of neighboring sentences using
# embeddings instead of splitting only by character count.
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load GOOGLE_API_KEY and other settings from a local .env file.
load_dotenv()

# GoogleGenerativeAIEmbeddings converts sentences into numeric vectors.
# A semantic breakpoint is created when neighboring sentences differ
# significantly according to the configured threshold.
text_splitter = SemanticChunker(
    GoogleGenerativeAIEmbeddings(
        model='gemini-embedding-2-preview'
    ),
    breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold_amount=1,
)

# These sentences contain several topic changes for the semantic splitter
# to detect.
sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

# Create Document objects so content and metadata can be carried together.
docs = text_splitter.create_documents([sample])

# The number of documents depends on embedding similarity and the threshold.
print(len(docs))
print(docs)