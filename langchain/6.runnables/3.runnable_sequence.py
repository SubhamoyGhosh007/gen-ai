from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableSequence

load_dotenv()

# Each prompt declares the input keys it expects.
prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model= 'gemini-2.5-flash')

# Convert the model's AIMessage response into plain text.
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(
    # RunnableSequence executes these stages from left to right.
    prompt1,
    model,
    parser,
    # Adapt the parser's string output to prompt2's expected dictionary input.
    RunnableLambda(lambda text: {'text': text}),
    prompt2,
    model,
    parser,
)

print(chain.invoke({'topic':'AI'}))

# These shorter chains demonstrate the output type after each stage.
c0 = prompt1 | model 
c1 = prompt1 | model | parser
c2 = prompt1 | model | parser | RunnableLambda(lambda text : {'text': text})
print(c0.invoke({'topic':'AI'}))
print(c1.invoke({'topic':'AI'}))
print(c2.invoke({'topic':'AI'}))