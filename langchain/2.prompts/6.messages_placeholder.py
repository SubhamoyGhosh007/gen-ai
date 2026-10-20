from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chat_history = []

file_name = Path(__file__).with_name('chat_history.txt')

with open(file_name) as f:
    chat_history.extend(f.readlines())

print(chat_history)

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': 'Where is my refund'
})
chat_history.extend(('human', 'Where is my refund'))

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
response = model.invoke(prompt)
# print(response.content)
chat_history.extend(AIMessage(response.content))
print(chat_history)