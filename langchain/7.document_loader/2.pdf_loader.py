# PyPDFLoader reads a PDF and creates one Document per page.
from langchain_community.document_loaders import PyPDFLoader

# The path points to the PDF that will be loaded.
loader = PyPDFLoader('/workspaces/gen-ai/langchain/7.document_loader/data/dl-curriculum.pdf')

# load() eagerly reads every page into a list.
docs = loader.load()

# The list length is the number of pages successfully extracted.
print(len(docs))

# page_content contains extracted text; metadata includes page and source information.
print(docs[0].page_content)
print(docs[0].metadata)