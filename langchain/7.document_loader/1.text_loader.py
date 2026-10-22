# TextLoader converts a text file into LangChain Document objects.
from langchain_community.document_loaders import TextLoader  
# ChatGoogleGenerativeAI sends the document text to Gemini for summarization.
from langchain_google_genai import ChatGoogleGenerativeAI
# StrOutputParser converts the model response into a plain string.
from langchain_core.output_parsers import StrOutputParser
# PromptTemplate keeps the prompt structure separate from the input poem.
from langchain_core.prompts import PromptTemplate
# Loads environment variables such as GOOGLE_API_KEY from a .env file.
from dotenv import load_dotenv 

load_dotenv()

# The model name is required by current versions of langchain-google-genai.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# {poem} is replaced when the chain receives {"poem": "..."}.
prompt = PromptTemplate(
    template= 'Write a summary for the following poem - \n {poem}',
    input_variables= ['poem']
)

parser = StrOutputParser()

# load() eagerly reads the file and returns a list of Document objects.
loader = TextLoader('/workspaces/gen-ai/langchain/7.document_loader/data/cricket.txt', encoding='utf-8')

docs = loader.load()

# TextLoader normally returns one Document containing the complete text file.
print(type(docs))
print(len(docs))

# Each Document stores its text in page_content and source details in metadata.
print(docs)
print(docs[0].page_content)

print(docs[0].metadata)

# The pipe operator creates a chain: prompt -> model -> string parser.
chain = prompt | model | parser

# invoke() supplies the value used to replace {poem} and runs the chain.
print(chain.invoke({'poem':docs[0].page_content}))