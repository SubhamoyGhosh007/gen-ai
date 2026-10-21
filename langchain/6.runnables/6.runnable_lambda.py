from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel, RunnableLambda

load_dotenv()

# RunnableLambda turns an ordinary Python function into a runnable stage.
def word_count(text):
    return len(text.split())

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model= 'gemini-3.6-flash')

parser = StrOutputParser()

# First generate and parse a joke, then calculate multiple values in parallel.
joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    # Preserve the original joke and derive a word count from the same input.
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'AI'})

# The parallel runnable returns a dictionary, so each branch is read by its key.
final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(final_result)
