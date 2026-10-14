from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
template = load_prompt(Path(__file__).with_name("template.json"))

prompt = template.invoke({
    'paper_input':"attention is all you need",
    'style_input':"technical",
    'length_input':"10 lines"
})
print(prompt)
response = model.invoke(prompt)
print(response.content)