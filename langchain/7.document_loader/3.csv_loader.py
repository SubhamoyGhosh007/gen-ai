# CSVLoader converts each CSV row into a LangChain Document.
from langchain_community.document_loaders import CSVLoader

# file_path identifies the CSV file to read.
loader = CSVLoader(file_path='/workspaces/gen-ai/langchain/7.document_loader/data/Social_Network_Ads.csv')

# load() eagerly reads all rows into a list of Documents.
docs = loader.load()

# The number of Documents normally matches the number of data rows.
print(len(docs))
# Each Document contains the row text and source/row metadata.
print(docs[1])