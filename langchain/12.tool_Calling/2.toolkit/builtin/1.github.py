import warnings

# Suppress warnings so the generated GitHub tools are easy to inspect.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper

load_dotenv()

# The wrapper reads GitHub credentials from the environment loaded above.
github = GitHubAPIWrapper(
    github_repository="langchain-ai/langchain"
)

toolkit = GitHubToolkit.from_github_api_wrapper(github)
# A toolkit groups related tools behind one provider-specific integration.
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name)
    print(tool.description)
    print()