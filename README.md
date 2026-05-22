# Slack Connector Agent

Bridges Slack users with the **agent orchestrator** network. Users send
instructions through Slack (DMs, @mentions, or slash commands); the agent
always forwards each request to `task-planner-agent` (`plan_task`) with a
Slack payload. The planner creates a workflow and forwards execution to
`task-executor-agent`.

Other agents in the network can also call the `send_slack_message` capability
exposed by this agent to push messages into any Slack channel.

---

## Architecture

```
Slack user
    │  (DM / @mention / /ask)
    ▼
Slack App  ─── Socket Mode ───►  slack_handler.py
                                        │
                                        │  discover_agents()
                                        │  send_task(plan_task)
                                        ▼
                              orchestrator_client.py
                                        │  WebSocket
                                        ▼
                              Agent Orchestrator (localhost:8000)
                                        │
                              ┌─────────┴─────────┐
                              ▼                   ▼
                    task-planner-agent    task-executor-agent
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| Agent orchestrator | running on `localhost:8000` |
| Slack workspace | admin access to create an app |

---

## Slack App Setup

### 1 — Create the app

1. Go to <https://api.slack.com/apps> and click **Create New App → From scratch**.
2. Give it a name (e.g. *AgentConnector*) and pick your workspace.

### 2 — Enable Socket Mode

Under **Settings → Socket Mode**, toggle **Enable Socket Mode** on.
Generate an **App-Level Token** with the scope `connections:write` — this is your `SLACK_APP_TOKEN` (`xapp-…`).

### 3 — Add Bot Token Scopes

Under **Features → OAuth & Permissions → Bot Token Scopes**, add:

| Scope | Why |
|-------|-----|
| `app_mentions:read` | Receive @mentions |
| `channels:history` | Read channel messages |
| `chat:write` | Post messages |
| `im:history` | Read DMs |
| `im:read` | Access DM metadata |
| `im:write` | Open DM channels |
| `commands` | Register slash commands |

### 4 — Register Slash Commands

Under **Features → Slash Commands**, add three commands:

| Command | Description | Usage hint |
|---------|-------------|------------|
| `/agents` | List connected agents | _(no hint needed)_ |
| `/ask` | Route an instruction | `<your instruction>` |
| `/agents-help` | Show help | _(no hint needed)_ |

### 5 — Subscribe to Events

Under **Features → Event Subscriptions**, enable events and subscribe to:

- `app_mention`
- `message.channels`
- `message.im`

### 6 — Install the App

Under **Settings → Install App**, click **Install to Workspace** and authorise.
Copy the **Bot User OAuth Token** (`xoxb-…`) — this is your `SLACK_BOT_TOKEN`.

### 7 — Copy the Signing Secret

Under **Settings → Basic Information → App Credentials**, copy the **Signing Secret** — this is your `SLACK_SIGNING_SECRET`.

---

## Local Setup

```bash
# Clone / enter the project
cd slack-connector-agent

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and fill in the three Slack values
```

### `.env`

```dotenv
ORCHESTRATOR_URL=http://localhost:8000

SLACK_BOT_TOKEN=xoxb-…
SLACK_APP_TOKEN=xapp-…
SLACK_SIGNING_SECRET=…

# Optional
TASK_TIMEOUT_MS=120000
```

---

## Running

Make sure the orchestrator is running first, then:

```bash
python main.py
```

You should see:

```
… Registered as <uuid>  ws_url=ws://localhost:8000/ws/<uuid>
… Slack Socket Mode handler started
… Slack Connector Agent is running. Press Ctrl+C to stop.
… WebSocket connected
```

---

## Using the Agent in Slack

### DM the bot

```
You: Find me the best noise-cancelling headphones under $300
Bot: ↻ Routing to task-planner-agent (capability: `plan_task`) …
Bot: ✅ task-planner-agent responded:
     Workflow planned and forwarded for execution …
```

### @mention in a channel

```
@AgentConnector summarise today's top Hacker News posts
```

### Slash commands

```
/agents              — list all connected agents
/ask <instruction>   — send instruction to task planner
/agents-help         — show full help
```

### Built-in text commands (in DMs / @mentions)

```
help          — show quick help
list agents   — show connected agents
```

---

## Exposing `send_slack_message` to Other Agents

Any agent in the orchestrator can discover the `slack-connector` and call
its `send_slack_message` capability:

```python
# Example: from another agent's code
result = await orchestrator.send_task(
    target_agent_id=slack_connector_id,
    capability="send_slack_message",
    input_data={
        "channel": "C01234567",          # channel ID
        "text": "Task finished! Here's the result …",
        "thread_ts": "1234567890.123456", # optional: reply in thread
    },
)
```

---

## Agent Settings (orchestrator dashboard)

The agent declares these settings at registration; operators can fill them
in via the orchestrator dashboard or REST API without restarting the agent:

| Key | Type | Description |
|-----|------|-------------|
| `slack_bot_token` | secret | Bot User OAuth Token |
| `slack_app_token` | secret | App-Level Token (Socket Mode) |
| `slack_signing_secret` | secret | Request verification secret |
| `default_channel` | string | Fallback channel for `send_slack_message` |
| `task_timeout_ms` | integer | Per-task timeout in milliseconds (default 120 000) |
