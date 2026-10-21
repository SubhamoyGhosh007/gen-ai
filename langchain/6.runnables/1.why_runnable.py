import random

# This small example mimics a traditional chain without using LangChain.
class FakeLLM:
    def __init__(self):
        print('LLM created')
    
    def predict(self, prompt):
        # A real LLM would use the prompt; this fake model returns a sample response.
        response_list = [
            'Delhi is the capital of India',
            'IPL is a cricket league',
            'AI stands for Artificial Intelligence'
        ]
        
        return {'response': random.choice(response_list)}
    

class FakePromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
        
    def format(self, input_dict):
        # Expand named placeholders using values from the input dictionary.
        return self.template.format(**input_dict)
    

class FakeLLMChain:
    def __init__(self, llm, template):
        self.llm = llm
        self.template = template
    
    def run(self, input_dict):
        # The chain connects prompt formatting to model execution.
        prompt = self.template.format(input_dict)
        result = self.llm.predict(prompt)
        
        return result['response']

template = FakePromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)

llm = FakeLLM()

chain = FakeLLMChain(llm, template)

response = chain.run(
    {'length':'short', 'topic': 'india'}
)
# The chain returns the parsed text rather than the model's response dictionary.
print(response)