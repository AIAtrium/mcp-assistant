from typing import Annotated, Any, Dict
import httpx
from arcade.sdk import ToolContext, tool
from arcade_exa.utils import EXA_API_CONFIG


@tool(requires_secrets=["EXA_API_KEY"])
async def github_search(
    context: ToolContext,
    query: Annotated[
        str, "Search query for GitHub repositories, or Github account, or code"
    ],
    num_results: Annotated[
        int, "Number of search results to return (default: 5)"
    ] = EXA_API_CONFIG["DEFAULT_NUM_RESULTS"],
) -> Annotated[dict, "A dictionary containing the Exa API search results for GitHub"]:
    """
    Search GitHub repositories using Exa AI - performs real-time searches on GitHub.com to find relevant repositories and GitHub accounts.
    """
    api_key = context.get_secret("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY is not set. Please provide a valid API key.")

    # Prefix the query if it doesn't already mention GitHub
    github_query = query if "github" in query.lower() else f"exa.ai GitHub: {query}"

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": api_key,
    }
    payload = {
        "query": github_query,
        "type": "auto",
        "includeDomains": ["github.com"],
        "numResults": num_results,
        "contents": {"text": True, "livecrawl": "always"},
    }
    url = str(EXA_API_CONFIG["BASE_URL"]) + str(EXA_API_CONFIG["ENDPOINTS"]["SEARCH"])

    async with httpx.AsyncClient(timeout=25) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if not data or (isinstance(data, dict) and not data.get("results")):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "No GitHub results found. Please try a different query.",
                    }
                ]
            }
        return {"content": [{"type": "text", "text": str(data)}]}
