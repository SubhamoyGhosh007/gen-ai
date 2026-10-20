# PydanticOutputParser validates generated fields against a Pydantic model.
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()

# Configure the deterministic chat model used for extraction.
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

# Field descriptions become format guidance for the model.
class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

# Parse generated text into a validated Person instance.
parser = PydanticOutputParser(pydantic_object=Person)

# Include parser instructions so the model emits compatible data.
template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# A LangChain expression chains prompt rendering, generation, and parsing.
chain = template | model | parser

response = chain.invoke({'place':'sri lankan'})

print(response)