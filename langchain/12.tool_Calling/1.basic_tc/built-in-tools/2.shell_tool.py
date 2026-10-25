import warnings

# Suppress warnings so the shell-tool example remains easy to read.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_community.tools import ShellTool

# ShellTool executes the supplied command in the current environment.
shell_tool = ShellTool()

results = shell_tool.invoke('ls')
print(results)