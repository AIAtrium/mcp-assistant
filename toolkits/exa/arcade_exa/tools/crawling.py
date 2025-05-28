from typing import Annotated, Any, Dict
import httpx
from arcade.sdk import ToolContext, tool
from arcade_exa.utils import EXA_API_CONFIG


@tool(requires_secrets=["EXA_API_KEY"])
async def crawling(
    context: ToolContext,
    url: Annotated[str, "The URL to crawl (e.g., 'exa.ai')"],
) -> Annotated[
    dict, "A dictionary containing the crawled content from the specified URL"
]:
    """
    Extract content from specific URLs using Exa AI - performs targeted crawling of web pages to retrieve their full content. Useful for reading articles, PDFs, or any web page when you have the exact URL. Returns the complete text content of the specified URL.
    """
    api_key = context.get_secret("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY is not set. Please provide a valid API key.")

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": api_key,
    }
    payload = {"ids": [url], "text": True, "livecrawl": "always"}
    crawl_url = str(EXA_API_CONFIG["BASE_URL"]) + "/contents"

    async with httpx.AsyncClient(timeout=25) as client:
        response = await client.post(crawl_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if not data or (
            isinstance(data, dict)
            and (not data.get("results") or len(data.get("results", [])) == 0)
        ):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "No content found at the specified URL. Please check the URL and try again.",
                    }
                ]
            }
        return {"content": [{"type": "text", "text": str(data)}]}
