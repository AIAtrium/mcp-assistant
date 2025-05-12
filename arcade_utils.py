from arcadepy import Arcade

AVAILABLE_TOOLS = {
    "tools":
    {
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
            "Google.GetThread"
        ],
        "Google Calendar": [
            "Google.CreateEvent",
            "Google.UpdateEvent",
            "Google.DeleteEvent",
            "Google.ListEvents",
            "Google.ListCalendars",
            "Google.FindTimeSlotsWhenEveryoneIsFree"
        ],
        "Notion": [
            "NotionToolkit.GetPageContentById",
            "NotionToolkit.GetPageContentByTitle",
            "NotionToolkit.CreatePage",
            "NotionToolkit.SearchByTitle",
            "NotionToolkit.GetObjectMetadata",
            "NotionToolkit.GetWorkspaceStructure",
        ]
    },
    "toolkits": [
        "github",
        "slack",
        "microsoft",
        "notion"
    ]
}

def get_tools_from_arcade(arcade_client: Arcade, provider: str):
    tools = []
    for toolkit in AVAILABLE_TOOLS["toolkits"]:
        tools.extend(arcade_client.tools.formatted.list(toolkit=toolkit, format=provider))

    for tool_list in AVAILABLE_TOOLS["tools"].values():
        for tool in tool_list:
            tools.append(arcade_client.tools.formatted.get(name=tool, format=provider))

    return tools
