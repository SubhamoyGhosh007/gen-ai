import streamlit as st
from langgraph_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid


# ************** Utils function
def generate_thread_id ():
    t_id = uuid.uuid4()
    return t_id

def reset():
    t_id = uuid.uuid4()
    st.session_state['thread_id'] = t_id
    st.session_state['message_history'] = []
    st.session_state['all_threads'].append(t_id)
    
def load_conversation(thread_id):
    state = chatbot.get_state(config = {
        'configurable':{
            'thread_id': thread_id
        }
    })
    return state.values.get('messages', [])

# ***************** Session State
if 'all_threads' not in st.session_state:
    st.session_state['all_threads'] = retrieve_all_threads()

if 'thread_id' not in st.session_state:
    uid = generate_thread_id()
    st.session_state['thread_id'] = uid
    st.session_state['all_threads'].append(uid)

if "message_history" not in st.session_state:
    st.title('New Conversation') 
    st.session_state.message_history = []
 

    

# ************** Sidebar
st.sidebar.title('Langgraph Chatbot')
if st.sidebar.button('New Chat'):
    reset()
st.sidebar.header('My Conversation')

if 'all_threads' in st.session_state:
    for thread_id in st.session_state['all_threads'][::-1]:
        if st.sidebar.button(str(thread_id)):
            st.session_state['thread_id'] = str(thread_id)
            messages = load_conversation(str(thread_id))
            
            temp_messages = []

            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = 'user'
                else:
                    role = 'assistant'
                temp_messages.append({'role': role, 'content': msg.content})
            st.session_state['message_history'] = temp_messages


for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.write(message['content'])
        
user_text = st.chat_input('Type your message...')

if user_text:
    with st.chat_message('user'):
        st.text(user_text)
    st.session_state['message_history'].append({
        'role':'user',
        'content': user_text
    })
    
    CONFIG = {
    'configurable': {
        'thread_id': st.session_state['thread_id']
    }
}
    
    with st.chat_message('assistant'):
        ai_response = st.write_stream(
            message_content for message_content, metadata in chatbot.stream({
                'messages': st.session_state['message_history']
            }, config=CONFIG, stream_mode='messages')
        )
    
    st.session_state['message_history'].append({
        'role': 'assistant',
        'content': ai_response
    })