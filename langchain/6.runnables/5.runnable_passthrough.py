from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel, RunnableLambda

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)


model = ChatGoogleGenerativeAI(model= 'gemini-2.5-flash')

parser = StrOutputParser()

# Generate one joke first, then preserve it while creating its explanation.
joke_gen_chain = RunnableSequence(prompt1, model, parser)

# RunnablePassthrough forwards the current value unchanged to the "joke" key.
# The lambda above supplies the dictionary shape required by prompt2.
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, RunnableLambda(lambda text: {'text': text}) ,parallel_chain)

# invoke executes the chain; the returned dictionary can then be indexed by key.
response = final_chain.invoke({'topic':'cricket'})

print(f"Joke: {response['joke']}")
print(f"Explanation: {response['explanation']}")