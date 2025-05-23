import os
from dotenv import load_dotenv
import pytest

# Import your tools
from arcade_exa.tools.linkedin_search import linkedin_search
from arcade_exa.tools.wikipedia_search import wikipedia_search_exa
from arcade_exa.tools.web_search import web_search_exa
from arcade_exa.tools.research_paper_search import research_paper_search
from arcade_exa.tools.github_search import github_search
from arcade_exa.tools.crawling import crawling
from arcade_exa.tools.competitor_finder import competitor_finder
from arcade_exa.tools.company_research import company_research

# Load .env for EXA_API_KEY
load_dotenv(dotenv_path="/Users/danmeier/Git/mcp-assistant/toolkits/exa/.env")

class MockContext:
    def get_secret(self, key):
        if key == "EXA_API_KEY":
            return os.environ.get("EXA_API_KEY")
        return None

@pytest.mark.asyncio
async def test_linkedin_search():
    context = MockContext()
    result = await linkedin_search(context, "Exa AI company page")
    assert "content" in result
    assert isinstance(result["content"], list)
    assert "Exa" in result["content"][0]["text"] or "exa" in result["content"][0]["text"]

@pytest.mark.asyncio
async def test_wikipedia_search_exa():
    context = MockContext()
    result = await wikipedia_search_exa(context, "Artificial intelligence")
    assert "content" in result
    assert isinstance(result["content"], list)
    assert "Wikipedia" in result["content"][0]["text"] or "wikipedia" in result["content"][0]["text"]

@pytest.mark.asyncio
async def test_web_search_exa():
    context = MockContext()
    result = await web_search_exa(context, "OpenAI")
    assert "content" in result
    assert isinstance(result["content"], list)
    assert "OpenAI" in result["content"][0]["text"] or "openai" in result["content"][0]["text"]

@pytest.mark.asyncio
async def test_research_paper_search():
    context = MockContext()
    result = await research_paper_search(context, "transformer neural networks")
    assert "content" in result
    assert isinstance(result["content"], list)
    assert "transformer" in result["content"][0]["text"].lower()

@pytest.mark.asyncio
async def test_github_search():
    context = MockContext()
    result = await github_search(context, "llama.cpp")
    assert "content" in result
    assert isinstance(result["content"], list)
    assert "github" in result["content"][0]["text"].lower()

@pytest.mark.asyncio
async def test_crawling():
    context = MockContext()
    # Use a simple, crawlable URL
    result = await crawling(context, "https://exa.ai")
    assert "content" in result
    assert isinstance(result["content"], list)
    assert "exa" in result["content"][0]["text"].lower()

@pytest.mark.asyncio
async def test_competitor_finder():
    context = MockContext()
    result = await competitor_finder(context, "web search API", exclude_domain="exa.ai", num_results=3)
    assert "content" in result
    assert isinstance(result["content"], list)
    assert "web" in result["content"][0]["text"].lower() or "search" in result["content"][0]["text"].lower()

@pytest.mark.asyncio
async def test_company_research():
    context = MockContext()
    result = await company_research(context, "exa.ai", subpages=1, subpage_target=["about", "pricing"])
    assert "content" in result
    assert isinstance(result["content"], list)
    assert "exa" in result["content"][0]["text"].lower()