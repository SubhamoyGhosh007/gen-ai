import warnings

# Suppress warnings so the schema and invocation are easier to inspect.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    # Field descriptions are exposed as part of the tool input schema.
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")
    
def multiply_func(a:int, b:int) -> int:
    return a*b

# StructuredTool adds an explicit name, description, and validated arguments.
multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema=MultiplyInput
)

result = multiply_tool.invoke({'a':3, 'b':3})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)