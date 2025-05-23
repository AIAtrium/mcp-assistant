from typing import Annotated
import httpx
from arcade.sdk import ToolContext, tool
from arcade_exa.utils import EXA_API_CONFIG

@tool(requires_secrets=["EXA_API_KEY"])
async def research_paper_search(
    context: ToolContext,
    query: Annotated[
        str,
        "Research topic or keyword to search for"
    ],
    num_results: Annotated[
        int,
        "Number of research papers to return (default: 5)"
    ] = EXA_API_CONFIG["DEFAULT_NUM_RESULTS"],
    max_characters: Annotated[
        int,
        "Maximum number of characters to return for each result's text content (Default: 3000)"
    ] = EXA_API_CONFIG["DEFAULT_MAX_CHARACTERS"],
) -> Annotated[
    dict,
    "A dictionary containing the Exa API search results for research papers"
]:
    """
    Search across 100M+ research papers with full text access using Exa AI - performs targeted academic paper searches with deep research content coverage. Returns detailed information about relevant academic papers including titles, authors, publication dates, and full text excerpts.
    """
    api_key = context.get_secret("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY is not set. Please provide a valid API key.")

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": api_key,
    }
    payload = {
        "query": query,
        "category": "research paper",
        "type": "auto",
        "numResults": num_results,
        "contents": {
            "text": {
                "maxCharacters": max_characters
            },
            "livecrawl": "fallback"
        }
    }
    url = EXA_API_CONFIG["BASE_URL"] + EXA_API_CONFIG["ENDPOINTS"]["SEARCH"]

    async with httpx.AsyncClient(timeout=25) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if not data or (isinstance(data, dict) and not data.get("results")):
            return {
                "content": [{
                    "type": "text",
                    "text": "No research papers found. Please try a different query."
                }]
            }
        return {
            "content": [{
                "type": "text",
                "text": str(data)
            }]
        }
