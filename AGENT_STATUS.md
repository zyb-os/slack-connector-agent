# Agent Status: NOT READY — NEEDS ATTENTION

**Status:** ❌ NOT READY TO ACCEPT TRAFFIC
**Reason:** Mandatory credentials are not configured

## Required Parameters (Missing)

| Parameter | Key | Description |
|-----------|-----|-------------|
| Slack Bot Token | `slack_bot_token` | Bot token from Slack app (format: `xoxb-…`) |
| Slack App-Level Token | `slack_app_token` | Socket Mode app token (format: `xapp-…`) |
| Slack Signing Secret | `slack_signing_secret` | Request signing secret from Slack app settings |

## How to Fix

1. Create a Slack app at https://api.slack.com/apps
2. Enable **Socket Mode** and generate an App-Level Token (`xapp-…`) with scope `connections:write`
3. Under **OAuth & Permissions**, add required bot scopes, install the app, and copy the Bot Token (`xoxb-…`)
4. Under **Basic Information → App Credentials**, copy the Signing Secret
5. Set the values via the orchestrator dashboard or in `.env`:

```env
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
SLACK_SIGNING_SECRET=<signing-secret>
```

This agent will remain offline and refuse traffic until all three required credentials are supplied.
