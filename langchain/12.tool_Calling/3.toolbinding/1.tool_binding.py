import warnings

# Suppress warnings so the model/tool message exchange is easier to follow.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import requests
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# A tool schema lets the model request this operation by name and arguments.
@tool
def multiply(a:int, b:int) -> int:
    """Given 2 numbers 1 and b, this tool returns their product or multplication"""
    return a*b

print(multiply.invoke({
    'a':3,'b':2
}))

# bind_tools enables the model to emit a structured tool call.
llm = ChatGroq(model='openai/gpt-oss-20b')
llm = llm.bind_tools([multiply])

query = HumanMessage('can you multply 3 with 21')

messages = [
    SystemMessage('You are a nice AI assistant'),
    query
]

response = llm.invoke(messages)
messages.append(response)

print(response)

# Execute the model's requested call, then return its result to the model.
tool_result = multiply.invoke(response.tool_calls[0])

messages.append(tool_result)
print(tool_result)

response = llm.invoke(messages)
print(response)
messages.append(response)
print('=' * 30)

for msg in messages:
    print(msg)
    print('_' * 20)
    