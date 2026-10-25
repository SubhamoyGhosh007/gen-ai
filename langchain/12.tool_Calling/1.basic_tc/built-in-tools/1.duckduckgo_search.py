import warnings

# Keep the lesson output focused on the tool result instead of dependency noise.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from langchain_community.tools import DuckDuckGoSearchRun

# DuckDuckGoSearchRun wraps search behind LangChain's standard tool interface.
search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke('top news in India today ?')

# Tool metadata is useful when inspecting what an agent can call.
print(results)
print(search_tool.name)
print(search_tool.description)
