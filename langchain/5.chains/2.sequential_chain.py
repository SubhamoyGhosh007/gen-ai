# Sequential LCEL stages pass the previous result into the next prompt.
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# First stage creates the detailed source text.
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

# Second stage consumes that text and creates a concise summary.
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# The model and parser are reused for both sequential stages.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

# Each pipe passes its output to the following Runnable.
chain = prompt1 | model | parser | prompt2 | model | parser

response = chain.invoke({'topic': 'Unemployment in India'})

print(response)

chain.get_graph().print_ascii()