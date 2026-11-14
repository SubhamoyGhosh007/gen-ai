import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from langgraph_backend import chatbot, return_all_threads


def generate_thread_id():
    return str(uuid.uuid4())


def load_thread(thread_id):
    state = chatbot.get_state(config={
        'configurable': {
            'thread_id': thread_id
        }
    })

    messages = []
    for message in state.values.get('messages', []):
        role = 'user' if isinstance(message, HumanMessage) else 'assistant'
        messages.append({'role': role, 'content': message.content})
    return messages


def start_new_chat():
    st.session_state['thread_id'] = generate_thread_id()
    st.session_state['message_history'] = []
    st.session_state['conversation_loaded'] = True


def load_conversation(thread_id):
    st.session_state['thread_id'] = str(thread_id)
    st.session_state['message_history'] = load_thread(str(thread_id))
    st.session_state['conversation_loaded'] = True


if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = None

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'conversation_loaded' not in st.session_state:
    st.session_state['conversation_loaded'] = False

if 'all_thread_ids' not in st.session_state:
    all_threads = return_all_threads()
    st.session_state['all_thread_ids'] = [
        str(thread_id) for thread_id in all_threads[::-1]
    ]


st.sidebar.title('Chatbot')

if st.sidebar.button('New Chat'):
    start_new_chat()
    st.rerun()

st.sidebar.header('My Conversations')

for thread_id in reversed(st.session_state['all_thread_ids']):
    if st.sidebar.button(str(thread_id)):
        load_conversation(thread_id)
        st.rerun()


if st.session_state['conversation_loaded']:
    st.title(str(st.session_state['thread_id']))
else:
    st.title('Select a conversation to continue')

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input = st.chat_input('Type your message here...')

if user_input:
    if not st.session_state['conversation_loaded']:
        start_new_chat()

    thread_id = st.session_state['thread_id']
    if thread_id not in st.session_state['all_thread_ids']:
        st.session_state['all_thread_ids'].append(thread_id)

    st.session_state['message_history'].append({
        'role': 'user',
        'content': user_input
    })

    config = {
        'configurable': {
            'thread_id': thread_id
        },
        'metadata':{
            'thread_id': thread_id
        },
        'run_name': f'chat_turn-{thread_id}'
    }
    
    response = st.write_stream(
                message_content.content
                for message_content, _ in chatbot.stream(
                    {"messages": st.session_state["message_history"]},
                    config=config,
                    stream_mode="messages",
                )
                if isinstance(message_content, AIMessage) and message_content.content
            )
    
    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': response
    })
    st.rerun()
