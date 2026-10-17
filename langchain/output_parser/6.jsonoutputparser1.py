from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()

class Facts(BaseModel):
    facts: list[str] = Field(
        description="Exactly five facts about the requested topic",
        min_length=5,
        max_length=5,
    )
    
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


parser = JsonOutputParser(pydantic_object=Facts)

template = PromptTemplate(
    template="""Give me exactly 5 facts about {topic}.

{format_instructions}""",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

chain = template | model | parser

result = chain.invoke({"topic": "black hole"})

print(result)