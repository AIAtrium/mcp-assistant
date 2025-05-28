from typing import List, Optional, Union, Literal
from pydantic import BaseModel

# Exa API Types

class ExaSearchRequest(BaseModel):
    query: str
    type: str
    category: Optional[str] = None
    includeDomains: Optional[List[str]] = None
    excludeDomains: Optional[List[str]] = None
    startPublishedDate: Optional[str] = None
    endPublishedDate: Optional[str] = None
    numResults: int
    contents: dict  # You can make this stricter if you want (see below)

class ExaCrawlRequest(BaseModel):
    ids: List[str]
    text: bool
    livecrawl: Optional[Literal['always', 'fallback']] = None

class ExaSearchResult(BaseModel):
    id: str
    title: str
    url: str
    publishedDate: str
    author: str
    text: str
    image: Optional[str] = None
    favicon: Optional[str] = None
    score: Optional[float] = None

class ExaSearchResponse(BaseModel):
    requestId: str
    autopromptString: str
    resolvedSearchType: str
    results: List[ExaSearchResult]

# Tool Types

class SearchArgs(BaseModel):
    query: str
    numResults: Optional[int] = None
    livecrawl: Optional[Literal['always', 'fallback']] = None

# If you want stricter typing for contents in ExaSearchRequest, use this:
class ExaSearchRequestContentsText(BaseModel):
    maxCharacters: Optional[int] = None

class ExaSearchRequestContents(BaseModel):
    text: Union[ExaSearchRequestContentsText, bool]
    livecrawl: Optional[Literal['always', 'fallback']] = None
    subpages: Optional[int] = None
    subpageTarget: Optional[List[str]] = None

# Then update ExaSearchRequest:
# contents: ExaSearchRequestContents
