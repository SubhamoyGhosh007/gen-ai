from langchain_core.runnables import RunnablePassthrough, RunnableParallel

def fake_llm(prompt: str) -> str:
    return 'completion'

# runnable = {
#     'llm1': fake_llm,
#     'llm2': fake_llm
# } | RunnablePassthrough.assign(
#     total_chars = lambda inputs: len(inputs['llm1'] + inputs['llm2'])
# )

runnable = RunnableParallel({
    'llm1' : fake_llm, 'llm2' : fake_llm}) | RunnablePassthrough.assign(
    total_chars = lambda inputs: len(inputs['llm1'] + inputs['llm2'])
)

result = runnable.invoke("hello")
print(result)