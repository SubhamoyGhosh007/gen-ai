from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter

from typing import TypedDict, Annotated, List
import sqlite3
from dotenv import load_dotenv

load_dotenv()

## LLM Model
llm = ChatGroq(model= 'openai/gpt-oss-120b')
# llm = ChatOpenRouter(model= 'cohere/north-mini-code:free')

## Database connection
conn =sqlite3.connect(database='chatbot.db', check_same_thread=False)

checkpointer =SqliteSaver(conn=conn)

## Defining state schema
class ChatMessages(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chat(state: ChatMessages) -> ChatMessages:
    messages = state['messages']
    response = llm.invoke(messages)
    print('Response:')
    print(response)
    return {'messages': [response]}
    
## Creating a graph
graph = StateGraph(ChatMessages)

# Creating node
graph.add_node('chat_node', chat)

# Creating edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

# retrieve all threads
def retrieve_all_threads():
    all_threads = set()
    for cp in checkpointer.list(None):
        all_threads.add(cp.config['configurable']['thread_id'])
    
    return list(all_threads)
