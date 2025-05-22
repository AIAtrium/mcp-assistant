import os
from dotenv import load_dotenv
import pytest
from arcade_exa.tools.linkedin_search import linkedin_search

load_dotenv(dotenv_path="/Users/danmeier/Git/mcp-assistant/toolkits/exa/.env")

@pytest.mark.asyncio
async def test_linkedin_search():
    # You need to pass a context, or mock it if required by your function
    # For now, let's assume you have a MockContext as before
    class MockContext:
        def get_secret(self, key):
            if key == "EXA_API_KEY":
                return os.environ.get("EXA_API_KEY")
            return None
    context = MockContext()
    result = await linkedin_search(context, "https://www.linkedin.com/in/dan-meier-/")
    # Adjust this line to match the actual return structure
    assert "Dan Meier" in result["content"][0]["text"]