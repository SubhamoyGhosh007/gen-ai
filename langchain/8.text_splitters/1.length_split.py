# CharacterTextSplitter can split text at an exact separator.
from langchain_text_splitters import CharacterTextSplitter
# PyPDFLoader reads each PDF page as a LangChain Document.
from langchain_community.document_loaders import PyPDFLoader

# Load the PDF and keep page content plus metadata in Document objects.
loader = PyPDFLoader(file_path='/workspaces/gen-ai/langchain/8.text_splitters/data/dl-curriculum.pdf')
docs = loader.load()

# An empty separator allows chunks to contain exactly 200 characters.
# There is no overlap, so each character belongs to only one chunk.
splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

# Split every loaded page into smaller Document objects.
result = splitter.split_documents(docs)

# Display the second generated chunk (index 1).
print(result[1].page_content)