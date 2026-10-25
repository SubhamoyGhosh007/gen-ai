import warnings
warnings.filterwarnings("ignore")

import json
import requests
from typing import Annotated
from dotenv import load_dotenv

from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_groq import ChatGroq
# Or: from langchain_google_genai import ChatGoogleGenerativeAI
# Or: from langchain_openrouter import ChatOpenRouter

load_dotenv()

# ==========================================
# STEP 1: DEFINE TOOLS WITH INJECTED ARG
# ==========================================

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Fetches the real-time conversion rate between two currencies."""
    url = f"https://open.er-api.com/v6/latest/{base_currency.upper()}"
    response = requests.get(url)
    data = response.json()
    return round(data["rates"][target_currency.upper()], 2)

@tool
def convert(
    base_currency_value: float, 
    conversion_rate: Annotated[float, InjectedToolArg]  # <--- Hidden from LLM!
) -> float:
    """Calculates the target currency value using a base amount and an injected conversion rate."""
    return round(base_currency_value * conversion_rate, 2)


# ==========================================
# STEP 2: DEMONSTRATE WHAT THE LLM ACTUALLY SEES
# ==========================================
print("=== 1. TOOL SCHEMA INSPECTION ===")
print("convert schema visible to the LLM:")
# Notice conversion_rate is stripped from the tool_call_schema!
print(json.dumps(convert.tool_call_schema.schema(), indent=2))

# Use a stable model that reliably generates parallel tool calls (Groq Llama-3.3, Gemini, etc.)
llm = ChatGroq(model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

query = "What is the conversion factor between USD and INR, and based on that convert 10 USD to INR?"
messages = [HumanMessage(content=query)]


# ==========================================
# STEP 3: LLM CALLS TOOLS (NO EXECUTION)
# ==========================================
print("\n=== 2. INVOKING MODEL (STEP 1) ===")
ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

print(f"AIMessage content: {repr(ai_message.content)}")
print(f"AIMessage tool_calls returned by LLM:")
print(json.dumps(ai_message.tool_calls, indent=2))

# DEMONSTRATION POINT:
# The model did NOT execute the API. It just output JSON dictionaries with arguments.
# Notice that for `convert`, the LLM only supplied `base_currency_value: 10`,
# completely unaware of `conversion_rate` because of InjectedToolArg!


# ==========================================
# STEP 4: DEVELOPER ACTS AS THE ORCHESTRATOR
# ==========================================
print("\n=== 3. MANUAL TOOL EXECUTION & INJECTION ===")

fetched_rate = None

for tool_call in ai_message.tool_calls:
    name = tool_call["name"]
    call_id = tool_call["id"]
    args = tool_call["args"]

    if name == "get_conversion_factor":
        print(f"-> Executing {name} with args: {args}")
        # Run tool locally
        fetched_rate = get_conversion_factor.invoke(args)
        print(f"   API result: {fetched_rate}")

        # Construct ToolMessage with the matching tool_call_id
        messages.append(ToolMessage(
            content=str(fetched_rate),
            tool_call_id=call_id,
            name=name
        ))

    elif name == "convert":
        print(f"-> Executing {name}")
        print(f"   LLM provided args: {args}")
        print(f"   Injecting conversion_rate = {fetched_rate}")

        # Inject the parameter manually before calling the Python function
        injected_args = {
            "base_currency_value": args["base_currency_value"],
            "conversion_rate": fetched_rate
        }
        
        calc_result = convert.invoke(injected_args)
        print(f"   Calculated result: {calc_result}")

        # Construct ToolMessage for convert
        messages.append(ToolMessage(
            content=str(calc_result),
            tool_call_id=call_id,
            name=name
        ))


# ==========================================
# STEP 5: RETURN CONTEXT BACK TO LLM
# ==========================================
print("\n=== 4. FINAL LLM INVOCATION WITH TOOL RESULTS ===")
# Now the model receives the conversation history:
# HumanMessage -> AIMessage (tool_calls) -> ToolMessage 1 -> ToolMessage 2
final_response = llm_with_tools.invoke(messages)

print(final_response.content)