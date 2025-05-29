from typing import Annotated, Any, Dict
import httpx
from arcade.sdk import ToolContext, tool
from arcade_exa.utils import EXA_API_CONFIG


@tool(requires_secrets=["EXA_API_KEY"])
async def linkedin_search(
    context: ToolContext,
    query: Annotated[
        str,
        "Search query for LinkedIn (e.g., <url> company page OR <company name> company page)",
    ],
    num_results: Annotated[
        int, "Number of search results to return (default: 5)"
    ] = EXA_API_CONFIG["DEFAULT_NUM_RESULTS"],
) -> Annotated[dict, "A dictionary containing the Exa API search results for LinkedIn"]:
    """
    Search LinkedIn for companies using Exa AI. Simply include company URL, or company name, with 'company page' appended in your query.
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
        "type": "auto",
        "includeDomains": ["linkedin.com"],
        "numResults": num_results,
        "contents": {
            "text": {"maxCharacters": EXA_API_CONFIG["DEFAULT_MAX_CHARACTERS"]}
        },
    }
    url = EXA_API_CONFIG["BASE_URL"] + EXA_API_CONFIG["ENDPOINTS"]["SEARCH"]

    async with httpx.AsyncClient(timeout=25) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if not data or (isinstance(data, dict) and not data.get("results")):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "No LinkedIn results found. Please try a different query.",
                    }
                ]
            }
        return {"content": [{"type": "text", "text": str(data)}]}
