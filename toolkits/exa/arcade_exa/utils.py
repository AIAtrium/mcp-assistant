from typing import TypedDict

class EndpointsDict(TypedDict):
    SEARCH: str

class ExaApiConfig(TypedDict):
    BASE_URL: str
    ENDPOINTS: EndpointsDict
    DEFAULT_NUM_RESULTS: int
    DEFAULT_MAX_CHARACTERS: int

EXA_API_CONFIG: ExaApiConfig = {
    "BASE_URL": "https://api.exa.ai",
    "ENDPOINTS": {
        "SEARCH": "/search"
    },
    "DEFAULT_NUM_RESULTS": 5,
    "DEFAULT_MAX_CHARACTERS": 3000
}
