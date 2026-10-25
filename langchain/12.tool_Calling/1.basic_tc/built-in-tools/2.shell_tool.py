import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_community.tools import ShellTool

shell_tool = ShellTool()

results = shell_tool.invoke('ls')
print(results)