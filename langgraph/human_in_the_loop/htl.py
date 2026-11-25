from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import tool
import requests
import os

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={os.getenv['ALPHA_VANTAGE_API_KEY']}"
    )
    r = requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }



def chat_node(state: ChatState)->ChatState:
    message = state['messages']
    response = llm.invoke(message)
    return {'messages': [response]}

llm = ChatGroq(model='openai/gpt-oss-120b')
all_tools = [get_stock_price, purchase_stock]
llm = llm.bind_tools(all_tools)

tool_node = ToolNode(all_tools)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

checkpointer = InMemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)

if __name__ == '__main__':
    thread_id = "demo-thread"
    
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break
        
        state = {"messages": [HumanMessage(content=user_input)]}
        
        result = chatbot.invoke(
            state,
            config={'configurable': {'thread_id': thread_id}}
        )

        interrupts = result.get("__interrupt__", [])
        if interrupts:
            # Our interrupt payload is the string we passed to interrupt(...)
            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("Your decision: ").strip().lower()

            # Resume graph with the human decision ("yes" / "no" / whatever)
            result = chatbot.invoke(
                Command(resume=decision),
                config={"configurable": {"thread_id": thread_id}},
            )

        # Get the latest message from the assistant
        messages = result["messages"] 
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content}\n")