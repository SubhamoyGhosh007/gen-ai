# A simple LCEL chain pipes prompt input through a model and parser.
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# PromptTemplate defines the chain's required input variable.
prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

# Configure the model and normalize its response to text.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

# LCEL's pipe operator composes compatible LangChain Runnables.
chain = prompt | model | parser

response = chain.invoke({'topic':'cricket'})

print(response)

chain.get_graph().print_ascii()