import os

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from typing import TypedDict, Annotated, List, NotRequired
from pydantic import BaseModel, Field
import sqlite3

from dotenv import load_dotenv

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'Chatbot Project'
os.environ["LANGCHAIN_TRACING_V2"] = "true"

## creating sqlite connection
conn = sqlite3.connect(database='langsmith_db', check_same_thread=False)

## SQLite Checkpointer
checkpointer = SqliteSaver(conn=conn)

## Creating LLM Chatbot Model
llm = ChatGroq(model="openai/gpt-oss-120b")

## Defining state schema
class ChatSchema(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

## defining the functionality of chat node
def chat_func(state: ChatSchema) -> ChatSchema:
    ai_response = llm.invoke(state['messages'])
    return {
        'messages': [ai_response]
    }

## creating a graph
graph = StateGraph(ChatSchema)

## Node creation
graph.add_node('chat_node', chat_func)

## Connecting the edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

def return_all_threads():
    all_threads = set()
    for cp in checkpointer.list(None):
        all_threads.add(cp.config['configurable']['thread_id'])
    return list(all_threads)

