from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['LANGCHAIN_PROJECT'] = 'simple_llm'

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

model = ChatGroq(model="openai/gpt-oss-120b")
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | model | parser

# Run it
result = chain.invoke({"question": "What is the capital of India?"})
print(result)
