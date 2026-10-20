# StrOutputParser normalizes a chat response to plain text.
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# Configure the model used for both report and summary prompts.
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

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

# Run the two prompts as separate model calls.
prompt1 = template1.invoke({'topic':'black hole'})
result = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result.content})
result1 = model.invoke(prompt2)

# Parse the final message content into a string.
parser = StrOutputParser()

# print(result.content)
# print(result1.content)
print(parser.invoke(result1.content))
