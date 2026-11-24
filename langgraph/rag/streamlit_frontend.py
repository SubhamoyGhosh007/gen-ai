from langgraph_backend_rag import (
    chatbot, 
    get_all_threads, 
    load_thread,
    ingest_pdf
    )
import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import pymupdf as fitz


st.title('Langgraph Chatbot')

## utilis functions
def get_thread_id():
    "It will return the current thread id or return"
    if 'thread_id' in st.session_state:
        return st.session_state['thread_id']
    else:
        return None

def create_new_chat():
    thread_id = str(uuid.uuid4())
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []
    return thread_id
   
def get_message_text(message):
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if text:
                    text_parts.append(text)
        
        return "\n".join(text_parts)
    return str(content)

def load_all_messages():
    if not st.session_state["thread_id"]:
        return
    
    else:
        st.session_state['message_history'] = []
        st.session_state['message_history'] = load_thread(st.session_state['thread_id'])
        for msg in st.session_state['message_history']:
            if isinstance(msg, HumanMessage):
                role = 'user'
            elif isinstance(msg, AIMessage):
                role = 'assitant'
            else:
                continue
            with st.chat_message(role):
                st.write(get_message_text(msg)) 
                
def load_conversation(thread_id):
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []
    # load_all_messages()

def show_all_conversations():
    st.session_state['all_threads'] = get_all_threads()
    if len(st.session_state['all_threads']) == 0:
        return
    for thread in st.session_state['all_threads'][::-1]:
        if st.sidebar.button(thread):
            load_conversation(thread)
            st.rerun()
    
def load_file():
    pass
      
## session informations
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = None

if 'ingested' not in st.session_state:
    st.session_state['ingested'] = set()

if 'all_threads' not in st.session_state:
    st.session_state['all_threads'] = None
    
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if st.session_state['thread_id'] is None:
    create_new_chat()
## sidebar ui
st.sidebar.title('Langgraph RAG Chatbot')
if st.sidebar.button('New Chat'):
    create_new_chat()
    st.rerun()
    
st.sidebar.header(
    f"Thread ID: {st.session_state['thread_id'] or ''}"
)


uploaded_file = st.sidebar.file_uploader(
    label='Choose a PDF.',
    type='pdf',
    key=f"uploader_{st.session_state['thread_id']}"
)

if uploaded_file is not None:
    file_key = (st.session_state['thread_id'], uploaded_file.name, uploaded_file.size)
    if file_key not in st.session_state['ingested']:
        doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
        text = "\n".join(
            page.get_text("text", sort=True)
            for page in doc
        )    
        if not st.session_state['thread_id']:
            thread_id = create_new_chat()
        else:
            thread_id = st.session_state['thread_id']
            
        docs_length = ingest_pdf(text=text, thread_id=thread_id)
    
st.sidebar.header('My Conversations')
show_all_conversations()
load_all_messages()

## main UI
user_text = st.chat_input('Type your message here..')
if user_text:
    if st.session_state['thread_id'] == None:
        create_new_chat()

    with st.chat_message("user"):
        st.write(user_text)
    
    CONFIG = {
        'configurable':{
            'thread_id': st.session_state['thread_id']
        }
    }
    
    with st.chat_message("assistant"):
        response = st.write_stream(
            get_message_text(message_data) for message_data, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_text)]},
                config=CONFIG,
                stream_mode='messages'
            ) if isinstance(message_data, AIMessage) and message_data.content
        )

        # print(response)
        # st.write(response)
    
    st.rerun()
