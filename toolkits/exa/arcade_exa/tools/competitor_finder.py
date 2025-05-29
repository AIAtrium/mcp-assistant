from typing import Annotated, Any, Dict, Optional
import httpx
from arcade.sdk import ToolContext, tool
from arcade_exa.utils import EXA_API_CONFIG


@tool(requires_secrets=["EXA_API_KEY"])
async def competitor_finder(
    context: ToolContext,
    query: Annotated[
        str,
        "Describe what the company/product in a few words (e.g., 'web search API', 'AI image generation', 'cloud storage service'). Keep it simple. Do not include the company name.",
    ],
    exclude_domain: Annotated[
        Optional[str],
        "Optional: The company's website to exclude from results (e.g., 'exa.ai')",
    ] = None,
    num_results: Annotated[int, "Number of competitors to return (default: 10)"] = 10,
) -> Annotated[
    dict, "A dictionary containing the Exa API search results for competitors"
]:
    """
    Find competitors of a company using Exa AI - performs targeted searches to identify businesses that offer similar products or services.
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
        "numResults": num_results,
        "contents": {
            "text": {"maxCharacters": EXA_API_CONFIG["DEFAULT_MAX_CHARACTERS"]},
            "livecrawl": "always",
        },
    }
    if exclude_domain:
        payload["excludeDomains"] = [exclude_domain]

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
                        "text": "No competitors found. Please try a different query.",
                    }
                ]
            }
        return {"content": [{"type": "text", "text": str(data)}]}
