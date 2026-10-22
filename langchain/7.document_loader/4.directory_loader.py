# DirectoryLoader finds files in a directory and delegates each file to a loader.
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    # This directory must exist and contain the PDF files to process.
    path='/workspaces/gen-ai/langchain/7.document_loader/data/books',
    # Only PDF files matching this pattern are selected.
    glob='*.pdf',
    # Each selected PDF is processed page by page.
    loader_cls=PyPDFLoader
)

# lazy_load() returns an iterator and loads Documents only when requested.
docs = loader.lazy_load()

# next() requests one Document at a time, avoiding an eager list of all pages.
f_doc = next(docs)
s_doc = next(docs)

# The iterator is stateful: the second next() returns the item after the first.
# print(f_doc.metadata)
# print(s_doc.metadata)