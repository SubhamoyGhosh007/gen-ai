import warnings

# Suppress warnings so the example output highlights the custom tool.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_community.tools import tool

# The docstring becomes the tool description visible to an agent.
@tool
def multiply(a: int, b:int) -> int:
    """Multiply two numbers"""
    return a*b

result = multiply.invoke({"a":3, "b":5})
print(result)