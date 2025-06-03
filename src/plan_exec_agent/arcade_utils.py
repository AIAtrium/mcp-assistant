import time
from enum import Enum

from arcadepy import Arcade


class ModelProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


AVAILABLE_TOOLS = {
    "tools": {
        "google": [
            "Google.SendEmail",
            "Google.SendDraftEmail",
            "Google.WriteDraftEmail",
            "Google.ListDraftEmails",
            "Google.ListEmailsByHeader",
            "Google.ListEmails",
            "Google.SearchThreads",
            "Google.ListThreads",
            "Google.GetThread",
            "Google.ChangeEmailLabels",
            "Google.CreateLabel",
            "Google.ListLabels",
            "Google.CreateEvent",
            "Google.UpdateEvent",
            "Google.ListEvents",
            "Google.ListCalendars",
        ],
        "NotionToolkit": [
            "NotionToolkit.GetPageContentById",
            "NotionToolkit.GetPageContentByTitle",
            "NotionToolkit.CreatePage",
            "NotionToolkit.SearchByTitle",
            "NotionToolkit.GetObjectMetadata",
            "NotionToolkit.GetWorkspaceStructure",
        ],
        "github":[
            "Github.CreateIssue",
            "Github.CreateIssueComment",
            "Github.ListPullRequests",
            "Github.GetPullRequest",
            "Github.UpdatePullRequest",
            "Github.ListPullRequestCommits",
            "Github.CreateReplyForReviewComment",
            "Github.ListReviewCommentsOnPullRequest",
            "Github.CreateReviewComment",
            "Github.ListOrgRepositories",
            "Github.GetRepository",
            "Github.ListRepositoryActivities",
        ],
        "slack": [
            "Slack.SendDmToUser",
            "Slack.SendMessageToChannel",
            "Slack.GetMembersInConversationById",
            "Slack.GetMembersInChannelByName",
            "Slack.GetMessagesInConversationById",
            "Slack.GetMessagesInChannelByName",
            "Slack.GetMessagesInDirectMessageConversationByUsername",
            "Slack.GetConversationMetadataById",
            "Slack.GetChannelMetadataByName",
            "Slack.GetDirectMessageConversationMetadataByUsername",
            "Slack.ListConversationsMetadata",
            "Slack.ListPublicChannelsMetadata",
            "Slack.ListPrivateChannelsMetadata",
            "Slack.ListGroupDirectMessageConversationsMetadata",
            "Slack.ListDirectMessageConversationsMetadata",
            "Slack.GetUserInfoById",
            "Slack.ListUsers",
        ],
        "microsoft": [
            "Microsoft.CreateEvent",
            "Microsoft.GetEvent",
            "Microsoft.ListEventsInTimeRange",
            "Microsoft.CreateDraftEmail",
            "Microsoft.UpdateDraftEmail",
            "Microsoft.SendDraftEmail",
            "Microsoft.CreateAndSendEmail",
            "Microsoft.ReplyToEmail",
            "Microsoft.ListEmails",
            "Microsoft.ListEmailsInFolder",
        ],
        "Hubspot": [
            "Hubspot.GetContactDataByKeywords",
            "Hubspot.CreateContact",
            "Hubspot.GetCompanyDataByKeywords"
        ],
        "Exa": [
            "Exa_Crawling",
            "Exa_CompanyResearch",
            "Exa_CompetitorFinder",
            "Exa_GithubSearch",
            "Exa_LinkedinSearch",
            "Exa_WikipediaSearchExa",
            "Exa_WebSearchExa",
            "Exa_ResearchPaperSearch",
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
    for tool_list in AVAILABLE_TOOLS["tools"].values():
        for tool in tool_list:
            try:
                tool_info = arcade_client.tools.formatted.get(name=tool, format=provider.value)
                tools.append(tool_info)
            except Exception as e:
                print(f"Error getting tool {tool}: {e}")
                continue
        time.sleep(3)

    return tools


def get_toolkits_from_arcade(
    arcade_client: Arcade,
    provider: ModelProvider,
    enabled_toolkits: list[str] | None = None,
):
    if not enabled_toolkits:
        print("WARNING: No toolkits enabled, using all available tools")
        return get_tools_from_arcade(arcade_client, provider)

    tools = []
    for toolkit in enabled_toolkits:
        if toolkit in AVAILABLE_TOOLS["tools"].keys():
            for tool in AVAILABLE_TOOLS["tools"][toolkit]:
                try:
                    tool_info = arcade_client.tools.formatted.get(name=tool, format=provider.value)
                    tools.append(tool_info)
                except Exception as e:
                    print(f"Error getting tool {tool}: {e}")
                    continue
            time.sleep(3)

    return tools
