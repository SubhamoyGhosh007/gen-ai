import warnings
# Keep the example output focused on the tool-calling flow.
warnings.filterwarnings("ignore")

import json
import requests
from typing import Annotated
from dotenv import load_dotenv

# InjectedToolArg marks values supplied by the application rather than the model.
from langchain_core.tools import tool, InjectedToolArg
# ToolMessage carries each executed tool result back into the conversation.
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openrouter import ChatOpenRouter

# Provider credentials are read from the local .env file.
load_dotenv()

# ========================================================
# STEP 1: DEFINE TOOLS WITH INJECTED ARGUMENT
# ========================================================

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Fetches the real-time conversion rate between two currencies."""
    # This tool performs the external lookup requested by the model.
    url = f"https://open.er-api.com/v6/latest/{base_currency.upper()}"
    response = requests.get(url)
    data = response.json()
    return round(data["rates"][target_currency.upper()], 2)

@tool
def convert(
    base_currency_value: float,
    conversion_rate: Annotated[float, InjectedToolArg]  # <--- Stripped from schema sent to the LLM!
) -> float:
    """Calculates the target currency value using a base amount and an injected conversion rate."""
    # The rate is calculated by the application and is not selected by the model.
    return round(base_currency_value * conversion_rate, 2)


# ========================================================
# STEP 2: DEMONSTRATE SCHEMA VISIBLE TO THE LLM
# ========================================================
print("=== 1. TOOL SCHEMA INSPECTION ===")
print("Schema of 'convert' exposed to the LLM (notice conversion_rate is missing):")
print(json.dumps(convert.tool_call_schema.model_json_schema(), indent=2))

llm = ChatOpenRouter(model="nvidia/nemotron-3-ultra-550b-a55b:free")
# Binding exposes only the model-visible schemas and lets the model request tools.
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

# Every model response and tool result is retained for the next model turn.
messages = [HumanMessage(content="What is the conversion factor between USD and INR, and based on that convert 10 USD to INR?")]

# Store the first tool's result so it can be injected into the second tool call.
saved_conversion_rate = None

# ========================================================
# STEP 3: MULTI-TURN ORCHESTRATION LOOP
# ========================================================
turn = 1
while True:
    print(f"\n=== TURN {turn}: INVOKING LLM ===")
    ai_response = llm_with_tools.invoke(messages)
    messages.append(ai_response)

    # If the LLM has stopped making tool calls, print final natural language answer and break
    if not ai_response.tool_calls:
        print("\n=== FINAL RESPONSE FROM LLM ===")
        print(ai_response.content)
        break

    # A response can contain multiple tool calls, so execute each one in order.
    print(f"LLM decided to call {len(ai_response.tool_calls)} tool(s):")
    for call in ai_response.tool_calls:
        print(f" -> Tool: {call['name']} | Args from LLM: {call['args']}")

    print("\n--- DEVELOPER EXECUTING TOOLS & INJECTING ARGS ---")
    for call in ai_response.tool_calls:
        name = call["name"]
        call_id = call["id"]
        args = call["args"].copy()

        if name == "get_conversion_factor":
            # Save the API result locally and send it back as a ToolMessage.
            saved_conversion_rate = get_conversion_factor.invoke(args)
            print(f"   [API Result] 1 {args['base_currency']} = {saved_conversion_rate} {args['target_currency']}")

            messages.append(ToolMessage(
                content=str(saved_conversion_rate),
                tool_call_id=call_id,
                name=name
            ))

        elif name == "convert":
            print(f"   [LLM Arg] Base currency value: {args['base_currency_value']}")
            print(f"   [Injected Arg] Injecting conversion_rate = {saved_conversion_rate}")

            # Rebuild the full runtime argument set, including the hidden value.
            injected_args = {
                "base_currency_value": args["base_currency_value"],
                "conversion_rate": saved_conversion_rate
            }

            calculated_result = convert.invoke(injected_args)
            print(f"   [Local Calculation Result] {calculated_result}")

            messages.append(ToolMessage(
                content=str(calculated_result),
                tool_call_id=call_id,
                name=name
            ))

    # The next turn lets the model consume the ToolMessages and continue.
    turn += 1