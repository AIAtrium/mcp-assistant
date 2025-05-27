import os
from dotenv import load_dotenv
from arcadepy import Arcade
from plan_exec_agent.arcade_utils import get_toolkits_from_arcade, ModelProvider
from pprint import pprint

# run from root: python -m src.plan_exec_agent.test_arcade_utils

load_dotenv()
api_key = os.getenv("ARCADE_API_KEY")

if not api_key:
    raise RuntimeError("ARCADE_API_KEY not set in environment or .env file")

# Replace with your actual Arcade API key or auth method
arcade_client = Arcade(api_key=api_key)

# Example: get all tools for OpenAI provider
tools = get_toolkits_from_arcade(arcade_client, ModelProvider.OPENAI)
print("\nAll OpenAI tools:")
pprint(tools)

# Example: get only Exa toolkit for Anthropic provider
tools_subset = get_toolkits_from_arcade(
    arcade_client,
    ModelProvider.ANTHROPIC,
    enabled_toolkits=["Exa"]
)
print("\nSubset tools (Anthropic, Exa only):")
pprint(tools_subset)