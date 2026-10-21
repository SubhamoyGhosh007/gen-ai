from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model= 'gemini-2.5-flash')

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(
    prompt1,
    model,
    parser,
    RunnableLambda(lambda text: {'text': text}),
    prompt2,
    model,
    parser,
)

print(chain.invoke({'topic':'AI'}))

# For understanding purpose only
c0 = prompt1 | model 
c1 = prompt1 | model | parser
c2 = prompt1 | model | parser | RunnableLambda(lambda text : {'text': text})
print(c0.invoke({'topic':'AI'}))
print(c1.invoke({'topic':'AI'}))
print(c2.invoke({'topic':'AI'}))