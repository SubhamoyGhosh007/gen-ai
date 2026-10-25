import warnings

# Suppress warnings so the agent/tool orchestration is easier to read.
warnings.filterwarnings("ignore")

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent

load_dotenv()

# Keep conversion self-contained with parameters visible to the model.
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


# 1. Initialize the model and tool list.
tools = [get_conversion_factor, convert]
llm = ChatOpenRouter(model="nvidia/nemotron-3-ultra-550b-a55b:free")

# 2. Build an agent that can choose and sequence these tools.
app = create_agent(llm, tools)

# 3. Invoke the agent with user input.
query = "What is the conversion factor between USD and INR, and please convert 10 USD to INR"
inputs = {"messages": [("user", query)]}

# The agent invokes tools, feeds their outputs back to the model, and stops
# when the model produces the final response.
final_state = app.invoke(inputs)

# 4. Extract the final response
final_message = final_state["messages"][-1]
print("=== Final Response ===")
print(final_message.content)