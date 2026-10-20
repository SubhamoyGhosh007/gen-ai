# Load a serialized PromptTemplate and use it like an in-code template.
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt

load_dotenv()

# Configure the chat model that consumes the rendered prompt.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# load_prompt reads the prompt definition from the companion JSON file.
template = load_prompt(Path(__file__).with_name("template.json"))

# Supply values for all template variables before model invocation.
prompt = template.invoke({
    'paper_input':"attention is all you need",
    'style_input':"technical",
    'length_input':"10 lines"
})
print(prompt)
response = model.invoke(prompt)
print(response.content)