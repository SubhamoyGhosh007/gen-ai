# Chat models accept messages or prompts and return message objects.
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load provider credentials from the local environment.
load_dotenv()

# Create a Gemini chat model with LangChain's chat interface.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# The response content is the generated assistant text.
response = model.invoke("Write a 5 line poem on cricket")
print(response.content)