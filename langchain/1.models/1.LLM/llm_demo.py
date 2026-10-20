# A text-generation model returns a plain string for a direct invocation.
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

# Load provider credentials from the local environment.
load_dotenv()

# Configure the Gemini-backed LangChain LLM wrapper.
llm = GoogleGenerativeAI(model="gemini-2.5-flash")

# Invoke the model with a natural-language prompt.
response = llm.invoke("What is the capital of India ?")

print(response)