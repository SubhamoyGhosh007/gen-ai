import warnings

# Suppress warnings so the BaseTool implementation remains the focus.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

# Pydantic defines the arguments accepted by the BaseTool implementation.
class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")
    
class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MultiplyInput

    # BaseTool calls _run when the tool is invoked synchronously.
    def _run(self, a: int, b: int) -> int:
        return a * b

multiply_tool = MultiplyTool()

result = multiply_tool.invoke({'a':3, 'b':3})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)

print(multiply_tool.args)
