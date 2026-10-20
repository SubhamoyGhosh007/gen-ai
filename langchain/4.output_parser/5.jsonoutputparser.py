# JsonOutputParser converts a model response into a Python JSON-compatible value.
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()

# Configure the model that will produce the requested facts.
model = ChatOllama(
    model="nemotron-3-nano:30b",
    base_url="https://ollama.com",
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"
        }
    },
    temperature=0
)

# The parser supplies instructions for valid JSON output.
parser = JsonOutputParser()

# Combine topic input with the parser's formatting instructions.
template = PromptTemplate(
    template='Give me 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# Chain prompt construction, model invocation, and JSON parsing.
chain = template | model | parser

response = chain.invoke({'topic':'black hole'})

print(response)