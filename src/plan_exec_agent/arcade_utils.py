from enum import Enum

from arcadepy import Arcade


class ModelProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


AVAILABLE_TOOLS = {
    "tools": {
        "Gmail": [
            "Google.SendEmail",
            "Google.SendDraftEmail",
            "Google.WriteDraftEmail",
            "Google.UpdateDraftEmail",
            "Google.DeleteDraftEmail",
            "Google.TrashEmail",
            "Google.ListDraftEmails",
            "Google.ListEmailsByHeader",
            "Google.ListEmails",
            "Google.SearchThreads",
            "Google.ListThreads",
            "Google.GetThread",
            "Google.ChangeEmailLabels",
            "Google.CreateLabel",
            "Google.ListLabels",
        ],
        "Google Calendar": [
            "Google.CreateEvent",
            "Google.UpdateEvent",
            "Google.DeleteEvent",
            "Google.ListEvents",
            "Google.ListCalendars",
            "Google.FindTimeSlotsWhenEveryoneIsFree",
        ],
        "Notion": [
            "NotionToolkit.GetPageContentById",
            "NotionToolkit.GetPageContentByTitle",
            "NotionToolkit.CreatePage",
            "NotionToolkit.SearchByTitle",
            "NotionToolkit.GetObjectMetadata",
            "NotionToolkit.GetWorkspaceStructure",
        ],
    },
    "toolkits": [
        "github",
        "slack",
        "microsoft",
        "google",
        "NotionToolkit",
        "Hubspot",
        "Exa",
    ],
}


def get_tools_from_arcade(arcade_client: Arcade, provider: ModelProvider):
    tools = []
    for toolkit in AVAILABLE_TOOLS["toolkits"]:
        tools.extend(
            arcade_client.tools.formatted.list(toolkit=toolkit, format=provider.value)
        )

    for tool_list in AVAILABLE_TOOLS["tools"].values():
        for tool in tool_list:
            tools.append(
                arcade_client.tools.formatted.get(name=tool, format=provider.value)
            )

    return tools


def get_toolkits_from_arcade(
    arcade_client: Arcade,
    provider: ModelProvider,
    enabled_toolkits: list[str] | None = None,
):
    if not enabled_toolkits:
        return get_tools_from_arcade(arcade_client, provider)

    tools = []
    for toolkit in enabled_toolkits:
        if toolkit in AVAILABLE_TOOLS["toolkits"]:
            tools.extend(
                arcade_client.tools.formatted.list(
                    toolkit=toolkit, format=provider.value
                )
            )

    return tools
