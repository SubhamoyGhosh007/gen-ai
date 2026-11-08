from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated, Literal, NotRequired
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

## LLM Model
llm = ChatGroq(model= 'openai/gpt-oss-120b')

checkpoint =InMemorySaver()

## Defining state schema
class ChatMessages(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat(state: ChatMessages) -> ChatMessages:
    messages = state['messages']
    response = llm.invoke(messages)
    print(response)
    return {'messages': [response]}
    
## Creating a graph
graph = StateGraph(ChatMessages)

# Creating node
graph.add_node('chat_node', chat)

# Creating edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpoint)

# response = chatbot.invoke({
#     'messages': [HumanMessage(content='Hi')]
# })

# print(response)