from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableParallel

load_dotenv()

# Both prompts receive the same input, but generate different kinds of content.
prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model= 'gemini-2.5-flash')

parser = StrOutputParser()

# RunnableParallel runs independent branches and combines their outputs by key.
parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

response = parallel_chain.invoke({'topic':'AI'})

# The result is a dictionary containing one value for each branch.
print(response)
print(response['tweet'])
print(response['linkedin'])
