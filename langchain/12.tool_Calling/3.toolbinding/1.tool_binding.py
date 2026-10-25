import warnings
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

@tool
def multiply(a:int, b:int) -> int:
    """Given 2 numbers 1 and b, this tool returns their product or multplication"""
    return a*b

print(multiply.invoke({
    'a':3,'b':2
}))

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
    