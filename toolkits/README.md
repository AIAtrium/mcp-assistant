# Toolkits

This directory contains the custom toolkits for the MCP Assistant. We host them in the Arcade Cloud.

**NOTE** This directory has its own virtual environment, `.venv/` and `requirements.txt` separate from any other modules around it. 

### How to run a toolkit locally

1. From the root of the toolkit name (i.e. `toolkits/exa`) run `arcade serve`
2. Visit http://localhost:8002/worker/health to see that your worker is running.
3. Run `ngrok http 8002` to get a public URL for your worker.
4. Navigate to the Workers https://api.arcade.dev/dashboard/workers page, click Add Worker, and fill in the form with the following values: ID: <toolkit\*name>, Worker Type: Arcade, URL: your public URL from ngrok/tailscale/cloudflare, **_Secret: dev_** and leave Timeout and Retry at default values, then click Create.
5. (if necessary) Add secrets in the Arcade Cloud for the worker: https://docs.arcade.dev/home/build-tools/create-a-tool-with-secrets

### How to add a new toolkit

https://docs.arcade.dev/home/build-tools/create-a-toolkit

1. Run in the toolkits directory: `arcade new`
2. `cd <toolkit_name>`
3. `make install`
4. Create a Python file for your tools in the `arcade_<toolkit_name>/tools` directory.

### How to deploy a new toolkit

1. Find the worker.toml file in your toolkit’s root directory or create one.
2. Add the following to the worker.toml file:

   ```[[worker]]

   [worker.config]
   id = "demo-worker"  # Choose a unique ID
   enabled = true
   timeout = 30
   retries = 3
   secret = "${env:WORKER_SECRET}"  # This is randomly generated for you by `arcade new`
   [worker.local_source]
   packages = ["./exa"]  # Path to your toolkit directory
   ```
3. Create and set an environmen variable called `WORKER_SECRET` with the value that `arcade new` generated for you
4. `arcade deploy`

# Tools

## Exa

The Exa toolkit is a collection of tools that use the Exa API. Based on this https://docs.exa.ai/examples/exa-mcp

| Tool Name               | Description                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| `company_research`      | Crawl and summarize company info from a website, including subpages and targeted sections.   |
| `competitor_finder`     | Find competitors for a company/product based on a description, with optional domain exclude. |
| `linkedin_search`       | Search LinkedIn for companies by name or URL.                                                |
| `research_paper_search` | Search 100M+ research papers by topic/keyword, returns detailed info and excerpts.           |
| `web_search_exa`        | General web search (real-time, can scrape content from URLs, configurable result count).     |
| `wikipedia_search_exa`  | Search Wikipedia specifically, returns relevant Wikipedia page content.                      |
| `crawling`              | Crawl a specific URL (article, PDF, etc.), returns the full text content.                    |
| `github_search`         | Search GitHub for repositories, accounts, or code (domain-limited to github.com).            |
