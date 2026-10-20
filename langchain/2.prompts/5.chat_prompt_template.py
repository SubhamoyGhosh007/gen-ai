# ChatPromptTemplate creates a structured prompt from role-based messages.
from langchain_core.prompts import ChatPromptTemplate

# Placeholders are filled when the template is invoked.
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

# Runtime values produce the final prompt messages.
prompt = chat_template.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)