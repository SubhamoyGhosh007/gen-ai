from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(page_title="Gemini Chatbot", page_icon="💬")
st.title("Gemini Chatbot")
st.caption("Ask questions and continue the conversation with Gemini.")
st.markdown(
    """
    <style>
    div[data-testid="stChatMessage"]:has(
        div[data-testid="stChatMessageAvatarUser"]
    ) {
        flex-direction: row-reverse;
    }

    div[data-testid="stChatMessage"]:has(
        div[data-testid="stChatMessageAvatarUser"]
    ) [data-testid="stChatMessageContent"] {
        width: 100%;
        align-items: flex-end;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful AI assistant.")
    ]

for message in st.session_state.chat_history:
    if isinstance(message, SystemMessage):
        continue
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

if user_input := st.chat_input("Type your message..."):
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = model.invoke(st.session_state.chat_history)
            response = str(result.content)
            st.markdown(response)
            st.session_state.chat_history.append(AIMessage(content=response))
        except Exception as error:
            st.error(f"Unable to get a response: {error}")
