from typing import Annotated, Any, Dict, Optional, List
import httpx
from arcade.sdk import ToolContext, tool
from arcade_exa.utils import EXA_API_CONFIG


@tool(requires_secrets=["EXA_API_KEY"])
async def company_research(
    context: ToolContext,
    query: Annotated[str, "Company website URL (e.g., 'exa.ai' or 'https://exa.ai')"],
    subpages: Annotated[
        Optional[int], "Number of subpages to crawl (default: 10)"
    ] = 10,
    subpage_target: Annotated[
        Optional[List[str]],
        "Specific sections to target (e.g., ['about', 'pricing', 'faq', 'blog']). If not provided, will crawl the most relevant pages.",
    ] = None,
) -> Annotated[
    dict, "A dictionary containing the Exa API search results for company research"
]:
    """
    Research companies using Exa AI - performs targeted searches of company websites to gather comprehensive information about businesses.
    """
    api_key = context.get_secret("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY is not set. Please provide a valid API key.")

    # Extract domain from query if it's a URL
    domain = query
    if "http" in query:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(query)
            domain = parsed.hostname.replace("www.", "") if parsed.hostname else query
        except Exception:
            pass

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": api_key,
    }
    contents = {
        "text": {"maxCharacters": EXA_API_CONFIG["DEFAULT_MAX_CHARACTERS"]},
        "livecrawl": "always",
        "subpages": subpages or 10,
    }
    if subpage_target:
        contents["subpageTarget"] = subpage_target

    payload = {
        "query": query,
        "category": "company",
        "includeDomains": [domain],
        "type": "auto",
        "numResults": 1,
        "contents": contents,
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
                        "text": "No company information found. Please try a different query.",
                    }
                ]
            }
        return {"content": [{"type": "text", "text": str(data)}]}
