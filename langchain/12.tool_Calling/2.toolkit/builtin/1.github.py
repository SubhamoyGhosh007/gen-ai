import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from dotenv import load_dotenv
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper

load_dotenv()

github = GitHubAPIWrapper(
    github_repository="langchain-ai/langchain"
)

toolkit = GitHubToolkit.from_github_api_wrapper(github)
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name)
    print(tool.description)
    print()