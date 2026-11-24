import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sqlite3
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.runnables import RunnableConfig
import faiss

load_dotenv()

## defining the chat state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
SYSTEM = SystemMessage(content=(
    "You are a helpful assistant. If the user asks about an uploaded document, "
    "call RagTool with a focused search query and answer from the retrieved text. "
    "If nothing relevant is returned, say so."
))
## defining the functionality of chat node
def chat_node(state: ChatState) -> ChatState:
    messages = state['messages']
    response = llm.invoke([SYSTEM] + messages)
    return {
        'messages': [response]
    }

## defining RAG as tool
@tool
def RagTool(query: str, config: RunnableConfig) -> str:
    """
    Search the PDF(s) the user uploaded in this conversation. Use this whenever
    the question may be answered by the uploaded document's contents.
    """
    thread_id = config.get('configurable', {}).get('thread_id')
    if not thread_id:
        return "No Conversation thread is available"
    retriever = vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.5,
            "filter": {
                'thread_id': thread_id
            }
        }
    )
    docs = retriever.invoke(query)
    if not docs:
        return "No documents have been uploaded for this conversation."
    return "\n\n".join(d.page_content for d in docs)

## chat model
llm = ChatGroq(model= "openai/gpt-oss-120b")
llm = llm.bind_tools([RagTool])

## embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name= 'sentence-transformers/all-MiniLM-L6-v2'
)


## defining the graph
graph = StateGraph(ChatState)

## defining the tool node
tool_node = ToolNode([RagTool])

# defining the nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

# defining the edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

## creating sql connection
conn = sqlite3.connect('rag_chatbot.db', check_same_thread=False)

## creating checkpointer
checkpointer = SqliteSaver(
    conn=conn
)

## compiling the graph
chatbot = graph.compile(checkpointer=checkpointer)

## returning all threads
def get_all_threads():
    all_threads = set()
    for cp in checkpointer.list(None):
        all_threads.add(cp.config['configurable']['thread_id'])
    return list(all_threads)

## returning the messages related to a thread
def load_thread(thread_id):
    CONFIG = {
        'configurable':{
            'thread_id': thread_id
        }
    }
    state = chatbot.get_state(config=CONFIG)
    return state.values.get('messages', [])

def create_empty_vector_store(embedding_model):
    # Get the embedding dimension (384 for all-MiniLM-L6-v2)
    dim = len(embedding_model.embed_query("dimension probe"))

    index = faiss.IndexFlatL2(dim)

    return FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={},
    )

vector_store = create_empty_vector_store(embedding_model)
## ingesting a pdf content to a thread
def ingest_pdf(text: str, thread_id: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    chunks = splitter.split_text(text)
    if not chunks:
        return 0
    documents = [
        Document(
            page_content=chunk,
            metadata={"thread_id": thread_id, "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]
    vector_store.add_documents(documents)
    return len(documents)
    
    