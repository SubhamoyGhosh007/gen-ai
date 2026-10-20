# JsonOutputParser can also use a Pydantic schema for field guidance.
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()

# Define an exact-length list constraint for the generated facts.
class Facts(BaseModel):
    facts: list[str] = Field(
        description="Exactly five facts about the requested topic",
        min_length=5,
        max_length=5,
    )
    
# Configure the deterministic model used for structured generation.
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


# Bind the Pydantic schema to JSON format instructions.
parser = JsonOutputParser(pydantic_object=Facts)

# Ask for facts and inject the parser's required JSON shape.
template = PromptTemplate(
    template="""Give me exactly 5 facts about {topic}.

{format_instructions}""",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

# The runnable sequence returns parsed JSON rather than raw model text.
chain = template | model | parser

result = chain.invoke({"topic": "black hole"})

print(result)