
import warnings

# Keep the lesson output focused on the tool result instead of dependency noise.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os

import requests
from dotenv import load_dotenv

# This is the legacy-compatible version. It uses langchain_classic so it can
# be compared with older AgentExecutor/ReAct tutorials.
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter

load_dotenv()

# The agent can search the web, call a weather API, and perform arithmetic.
search_tool = DuckDuckGoSearchRun()


@tool
def multiply(a: float, b: float) -> float:
    """Return the product of two numbers."""
    return round(a * b, 2)


@tool
def get_weather_data(city: str) -> str:
    """Fetch the current weather for a city."""
    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "WEATHERSTACK_API_KEY is not set. Add it to your .env file."
        )

    response = requests.get(
        "https://api.weatherstack.com/current",
        params={
            "access_key": api_key,
            "query": city,
        },
        timeout=15,
    )

    response.raise_for_status()
    return response.text


tools = [
    search_tool,
    get_weather_data,
    multiply,
]

llm = ChatOpenRouter(
    model="nvidia/nemotron-3-ultra-550b-a55b:free"
)

# Keep the ReAct prompt local. This avoids pulling and deserializing a public
# LangSmith prompt, which current LangChain versions block by default.
prompt = PromptTemplate.from_template(
    """Answer the following question as accurately as possible.
You have access to the following tools:

{tools}

Use this format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the answer to the original question

Question: {input}
Thought: {agent_scratchpad}"""
)

# AgentExecutor accepts the legacy {"input": "..."} request shape and returns
# the final text in response["output"].
react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

response = agent_executor.invoke(
    {
        "input": (
            "Find the capital of Madhya Pradesh, then find its "
            "current weather condition. Then multiply the temperature by 11."
        )
    }
)

print(response["output"])