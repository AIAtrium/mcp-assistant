# MCP Agent

This is an AI assistant that runs using MCP. 

## How to install before running
### Basic Setup
First create a python venv then `pip install -r requirements.txt`.

This application uses [Arcade](https://www.arcade.dev/) to facilitate tool use so you don't need to run any local MCP servers or connect to any remotely in the cloud. Please sign up for Arcade and get [a free api key here](https://api.arcade.dev/dashboard/api-keys) once registered to use this repo. 

Set the following enviroment variables in an `.env` file in the project root:
```
ANTHROPIC_API_KEY=
ARCADE_API_KEY=
OPENAI_API_KEY=
```
You **only need ONE LLM key**. This repo only supports OpenAI and Anthropic models currently. 

By default the agent will kick off an OAuth flow with Arcade for any tools that its not authorized to use. This will cause the exectuion to pause until that flow completes. If you want to skip any unauthorized tools without kicking off OAuth so that it doesn't block, set `SKIP_CLI_AUTH=true`.

### Observability and tracing
This project uses [Langfuse](https://github.com/langfuse/langfuse) for observability and tracing. Its not mandatory but its helpful to see the dozens of LLM calls that occur nicely formatted in a UI. To utilize it, set these environment variables in the `.env`:

```
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_HOST="https://cloud.langfuse.com"
```

## How to Run
The MCP Assistant application is heavily customizable. You can do anything email triaging to lead qualification.

The default workflow generates a **daily briefing** over various tools like email, calendar, whatsapp and Notion that tells you your schedule for the day, gives you background research from the internet about who you're meeting with and tells you what is urgent. 

### Run the Default Flow - Daily Briefing in Notion
1. Create a Notion page called `Daily Briefings`. Ideally this should be at the root of your Notion workspace.   
2. Update the `user_context` in `main` in `main.py` with a description of who you are, how you like to work (like what communication tools you prefer) and any extra information that will make the model's output better. Update the `base_system_prompt` with any special instructions you want it to take into account. 
3. The instructions for the LLM on *how* to generate the daily briefing are in the  `INPUT_ACTION` variable in `main.py` in the `main` method. Change the step-by-step instructions based on how you want you daily briefing created.  
For example, if you want the briefing to also check a specific Notion page that has your tasks, the model can also do this
4. (Optional) You can alter the `DEFAULT_TOOLS` array at the top of `plan_exec_agent
.py` to have only these values `["Gmail", "Google Calendar", "Whatsapp", "Exa", "Notion"]` so you don't have to set up every MCP server
5. run `python main.py` to create a daily briefing one time. 

### Run your own Custom Flow
Create a file called `user_inputs.py` in root directory then add the following constant variables that will control how the assistant runs:
1. `INPUT_ACTION` - your request for the MCP assistant
2. `USER_CONTEXT` - background information on who you are, what your preferences are, how you like to work, etc. Helps the model produce better output
3. `BASE_SYSTEM_PROMPT` - any special instructions the model may need to take into account beyond your user preferences. defaults to 'you are a helpful assistant'
4. `ENABLED_CLIENTS` - an array of strings containing the names of the MCP servers you want the system to have access to. Make sure they are the same names as those used in the `mcp_clients/` implementations. This will allow you to omit, for example, outlook, and still have the system run without issue

Then run `python main.py` to see the result

### Linking the Results to Other Systems using a Queue
This project supports publishing the `state` during the agent's main execution loop using Redis.


**To run Redis locally, use:**
```
docker run -d --name redis-local -p 6379:6379 redis:latest
```

Please add the following environment variables to your .env file
```
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_STREAM_NAME=plan_execution
PUBLISH_TO_REDIS=true
```

If you need to debug redis, use the following REDIS CLI commands:
First run this to start a terminal in the container `docker exec -it redis-local redis-cli`
Then
```
KEYS * # check streams
XLEN plan_execution # check this specific stream
XREVRANGE plan_execution + - COUNT 1  # check the last message
```

## Future Work

- More connectors
- Script to run the briefing automatically at the same time every day as a background process
- Fix asyncio log pollution
- Make sure the agent can iterate over every item for a given task (i.e. all 50 emails in an inbox, rather than only the first 10)
- Save the execution graph after a successful run and load it next time instead of regenerating it
- Context window reduction mechanism / token limiting to save money
- Configurability for different models at different steps - planning vs. executing, cheap vs. expensive
- Support more models

## Instructions on How to Run the Legacy MCP Functionality

You must install MCP servers in order to use this application. You do not need to install them all, the code will run without them all installed but their tools will not be accessible. The instructions for all of them are below:

1. create a python venv then `pip install -r requirements.txt`
2. You need to `git  clone` the following repos and run `npm install` in all the project roots
    - https://github.com/nspady/google-calendar-mcp
    - https://github.com/GongRzhe/Gmail-MCP-Server
    - https://github.com/makenotion/notion-mcp-server
    - https://github.com/exa-labs/exa-mcp-server
    - https://github.com/AIAtrium/outlook-mcp
    - https://github.com/AIAtrium/slack-mcp-server
3. Follow the instructions at https://github.com/nspady/google-calendar-mcp to set up OAuth for google
   1. enable Gmail and Google Calendar APIs on Google Cloud
   2. create an OAuth app for desktop
   3. download the JSON credentials and change the name to `gcp-oauth.keys.json`, then copy it to the root of both the gmail and gcal projects from above
   4. run `npm run auth` from the root of `google-calendar-mcp` to set up OAuth access
4. For the **Gmail MCP Server** follow the instructions in the README under _"2. Run Authentication"_  
   TLDR: Place the `gcp-oauth.keys.json` file into the root of **Gmail MCP Server** project OR run `mkdir -p ~/.gmail-mcp && mv gcp-oauth.keys.json ~/.gmail-mcp/`. Run `npm run auth` from the root of **Gmail-MCP-Server** to enable OAuth
5. Follow the instructions at https://github.com/makenotion/notion-mcp-server to set up Notion access
6. Get an [Exa API key here](https://dashboard.exa.ai/api-keys). 
7. WhatsApp support - Follow the instructions at https://github.com/lharries/whatsapp-mcp to clone and install. This will require you to authenticate to Whatsapp at least once via QR code **in the terminal**
    1. This requires a specific version of `golang`. If you have already installed go (i.e. with homebrew) and don't want to override it for fear of breaking some system dependcies, download `gvm` then use it to download the required version. You may need to configure your `$PATH` to account for different go versions via gvm
    2. This repo only has instructions to run via Claude Desktop but you can use the `uv` package manager to create a `.venv` inside the `whatsapp-mcp-server` directory and install the dependencies. These are different than the dependencies for this project. The `host` in this project will automatically activate the venv for you when you run this project
    3. You will need to regularly run the `main.go` script to sync the latest messages into the sqlite DB.
8. Outlook - you need to follow the instructions in the [README](https://github.com/AIAtrium/outlook-mcp) to set up the Azure app with the correct scopes. This is our fork of an [open source MCP server](https://github.com/ryaker/outlook-mcp), which we altered to prevent having the same tool names as the gmail MCP server and to make sure that the Azure token is refreshed automatically 
Once that is done, do the following
    1. run `npm start` in a terminal
    2. open another terminal and run `npm run auth-server`
    3. visit [http://localhost:3333/auth](http://localhost:3333/auth) to authenticate. This will create a `.outlook-mcp-tokens.json` file in your `$HOME` directory. Do NOT go to http://localhost:3333/auth/callback, it will error
    *Note* you may have to reauthenticate
9. Get the Slack User Token, follow the instructions at https://github.com/AIAtrium/slack-mcp-server to configure the scope correctly. Then you must manually add the bot to each public, private and DM channel you want it to have access to using `/invite @your-bot-name`
10. (Optional) You may want to comment out code in the above repos that expose tools such as ones that enable delete of emails or notion pages. You may also want to grant less privileges (scopes) in Google Cloud and Azure
11. Set the following enviroment variables in an `.env` file in the project root:
```
ANTHROPIC_API_KEY=
GCAL_MCP_SERVER_PATH=/path/to/google-calendar-mcp/build/index.js
GMAIL_MCP_SERVER_PATH=/path/to/Gmail-MCP-Server/dist/index.js
OPENAPI_MCP_HEADERS={"Authorization": "Bearer ntn_YOUR_TOKEN_HERE", "Notion-Version": "2022-06-28"}
NOTION_MCP_SERVER_PATH=/path/to/notion-mcp-server/bin/cli.mjs
WHATSAPP_MCP_SERVER_PATH=/path/to/whatsapp-mcp/whatsapp-mcp-server/main.py
WHATSAPP_MCP_SERVER_VENV_PATH=/path/to/whatsapp-mcp/whatsapp-mcp-server/.venv
EXA_API_KEY=
EXA_MCP_SERVER_PATH=/path/to/exa-mcp-server/build/index.js
OUTLOOK_MCP_SERVER_PATH=/path/to/outlook-mcp/index.js
MS_CLIENT_ID=
MS_CLIENT_SECRET=
SLACK_MCP_SERVER_PATH=/path/to/slack-mcp-server/dist/index.js
SLACK_USER_TOKEN=
```

**Note** the unusual setup for the _Notion Token_ under `OPENAPI_MCP_HEADERS`