import warnings

# Suppress warnings so the toolkit members and their results stand out.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_core.tools import tool

# Each decorated function becomes a tool that can be included in the toolkit.
@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

class MathToolkit:
    def get_tools(self):
        # Returning a list keeps the toolkit compatible with agent APIs.
        return [add, multiply]

toolkit = MathToolkit()
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name, "=>", tool.description)
    print(tool.invoke({'a': 3 , 'b' : 6}))
