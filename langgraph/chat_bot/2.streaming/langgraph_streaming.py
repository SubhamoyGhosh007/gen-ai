from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from typing import TypedDict, Annotated, List, Literal

from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model='openai/gpt-oss-120b')

checkpoint = InMemorySaver()
config = {
    'configurable': {
        'thread_id': 'thread-id-1'
    }
}

class ChatState(TypedDict):
    current_user_text: str
    messages: Annotated[List[BaseMessage], add_messages]

def user_text(state: ChatState) -> ChatState:
    print()
    user_text = input('User: ')
    return {'current_user_text': user_text, 'messages': [HumanMessage(user_text)]}

def chat_node(state: ChatState) -> ChatState:
    messages = state['messages']
    print('AI: ', end='', flush=True)
    response = llm.invoke(messages)
    return {'messages': [response]}

def check_user_message(state: ChatState) -> Literal['yes', 'no']:
    current_user_text = state['current_user_text']
    if current_user_text.lower() == 'exit':
        return 'yes'
    return 'no'

graph = StateGraph(ChatState)

graph.add_node('user_text', user_text)
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'user_text')
graph.add_conditional_edges(
    'user_text', 
    check_user_message,
    {
        'yes': END,
        'no': 'chat_node'
    }
)
graph.add_edge('chat_node', 'user_text')

chatbot = graph.compile(checkpointer=checkpoint)

initial_state: ChatState = {
    'messages': [SystemMessage('You are a helpful assistant.')]
}

# Stream the graph and filter out user/system messages from printing
for message_chunk, metadata in chatbot.stream(
    initial_state,
    config=config,
    stream_mode='messages'
):
    if isinstance(message_chunk, AIMessage) and message_chunk.content:
        print(message_chunk.content, end='', flush=True)
    