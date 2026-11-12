import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "agent"

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  api_key = os.getenv("WEATHERSTACK_API_KEY")
  if not api_key:
    raise RuntimeError("WEATHERSTACK_API_KEY is not set. Add it to your .env file.")

  response = requests.get(
    "https://api.weatherstack.com/current",
    params={"access_key": api_key, "query": city},
    timeout=15,
  )
  response.raise_for_status()
  return response.text

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Keep the ReAct prompt local instead of pulling a serialized public prompt
# from LangSmith.
prompt = PromptTemplate.from_template(
  """Answer the following question as accurately as possible. You have access
to the following tools:

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

# Step 3: Create the ReAct agent with the local prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# Step 5: Invoke
response = agent_executor.invoke({"input": "What is the current temp of gurgaon"})
print(response)

print(response['output'])