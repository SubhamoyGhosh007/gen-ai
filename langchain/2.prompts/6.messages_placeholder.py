# MessagesPlaceholder injects prior conversation turns into a chat prompt.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Start with history loaded from a simple companion text file.
chat_history = []

file_name = Path(__file__).with_name('chat_history.txt')

with open(file_name) as f:
    chat_history.extend(f.readlines())

print(chat_history)

# Keep history as a dedicated prompt variable between user turns.
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# Render the prompt with the current query and prior messages.
prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': 'Where is my refund'
})
# Add the new user turn before requesting the assistant response.
chat_history.extend(('human', 'Where is my refund'))

# Invoke the model and append its AI message to the history.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
response = model.invoke(prompt)
# print(response.content)
chat_history.extend(AIMessage(response.content))
print(chat_history)