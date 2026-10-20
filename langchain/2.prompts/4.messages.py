# Chat history is represented explicitly with typed LangChain messages.
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


# Seed the conversation with system guidance and a user message.
messages = [
    SystemMessage(content="You are a helpful assitant."),
    HumanMessage(content="Tell me about LangChain")
]

# Chat models consume the message list and return an AIMessage.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

result = model.invoke(messages)

# Append the assistant turn to preserve the conversation history.
messages.append(AIMessage(content=result.content))

print(messages)