from abc import ABC, abstractmethod
import random

# A Runnable accepts input and returns output through a common invoke method.
class Runnable(ABC):
    
    @abstractmethod
    def invoke(input_data):
        pass
    
class FakeLLM(Runnable):
    def __init__(self):
        print('LLM created')
    
    def predict(self, prompt):
        response_list = [
            'Delhi is the capital of India',
            'IPL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]
        
        return {'response': random.choice(response_list)}
    
    def invoke(self, prompt):
            # invoke is the standard entry point used by a runnable chain.
            response_list = [
                'Delhi is the capital of India',
                'IPL is a cricket league',
                'AI stands for Artificial Intelligence'
            ]
            
            return {'response': random.choice(response_list)}
        
class FakePromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
        
    def format(self, input_dict):
        return self.template.format(**input_dict)
    
    def invoke(self, input_dict):
        # This runnable converts structured input into a formatted prompt.
        return self.template.format(**input_dict)
        
class FakeStrOutputParser(Runnable):
    def __init__(self):
        pass
     
    def invoke(self, input_data):
        # Extract the text value from the fake model's response dictionary.
        return input_data['response']
    
class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list
        
    def invoke(self, input_data):
        # Pass each stage's output as the next stage's input.
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        
        return input_data
    
    
llm = FakeLLM()
template = FakePromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)
parser = FakeStrOutputParser()

# print(llm.invoke(template.invoke({'length':'short', 'topic': 'india'}))['response'])

chain = RunnableConnector(runnable_list=[template, llm, parser])
# The result flows through template -> model -> parser.
response = chain.invoke({'length':'long', 'topic':'india'})

print(response)

template1 = FakePromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

template2 = FakePromptTemplate(
    template='Explain the following joke {response}',
    input_variables=['response']
)

llm1 = FakeLLM()
parser1 = FakeStrOutputParser()

chain1 = RunnableConnector([template1, llm1])
chain2 = RunnableConnector([template2, llm1, parser1])

final_chain = RunnableConnector([chain1, chain2])

# Nested connectors show that complete chains can be used as runnable stages.
final_response = final_chain.invoke({'topic':'cricket'})
print(final_response)

