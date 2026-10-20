# StructuredOutputParser turns model text into named fields from a schema.
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
import os

load_dotenv()

# Configure the hosted Ollama-compatible chat model.
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

# ResponseSchema supplies field names and guidance for the model.
schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

# Generate format instructions and parse the model's structured response.
parser = StructuredOutputParser.from_response_schemas(schema)

# Inject parser instructions into a reusable prompt template.
template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# Invoke the prompt, then convert the returned text into a dictionary.
prompt = template.invoke({
    'topic': 'black hole'
})

response = model.invoke(prompt)

result = parser.invoke(response.content)

print(result)