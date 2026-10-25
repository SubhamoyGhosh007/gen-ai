import warnings
warnings.filterwarnings("ignore")

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Keep convert self-contained with its parameters visible to the LLM
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Fetches the currency conversion factor between a given base currency and a target currency."""
    url = f"https://open.er-api.com/v6/latest/{base_currency.upper()}"
    response = requests.get(url)
    data = response.json()
    return round(data["rates"][target_currency.upper()], 2)

@tool
def convert(base_currency_value: float, conversion_rate: float) -> float:
    """Calculates the target currency value given a base amount and the conversion rate."""
    return round(base_currency_value * conversion_rate, 2)


# 1. Initialize Model and Tool list
tools = [get_conversion_factor, convert]
llm = ChatOpenRouter(model="nvidia/nemotron-3-ultra-550b-a55b:free")

# 2. Build the autonomous ReAct agent
app = create_react_agent(llm, tools)

# 3. Invoke with user input
query = "What is the conversion factor between USD and INR, and please convert 10 USD to INR"
inputs = {"messages": [("user", query)]}

# The agent autonomously invokes the model, runs the tools, appends tool outputs,
# feeds them back to the model, and finishes when the model produces the final text.
final_state = app.invoke(inputs)

# 4. Extract the final response
final_message = final_state["messages"][-1]
print("=== Final Response ===")
print(final_message.content)