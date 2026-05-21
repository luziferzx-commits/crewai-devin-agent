# CrewAI → Devin Project Manager 🤖

A **Multi-Agent system** powered by [CrewAI](https://crewai.com) that translates simple Thai-language requests into detailed technical task specifications, then automatically triggers a new [Devin](https://devin.ai) session to execute each task.

## Architecture

```
┌──────────────┐         ┌─────────────────────┐         ┌──────────────┐
│  User (Thai) │──────▶  │  Project Manager     │──────▶  │  Devin API   │
│  "เพิ่มปุ่ม…" │         │  Agent (CrewAI)      │         │  v3 REST     │
└──────────────┘         │  - translate TH→EN   │         │  New Session │
                         │  - expand to spec    │         └──────────────┘
                         │  - dispatch via tool  │
                         └─────────────────────┘
```

### Components

| File | Description |
|---|---|
| `main.py` | CLI entry-point (interactive & argument modes) |
| `agents.py` | Project Manager Agent definition |
| `tasks.py` | Task template: translate → expand → dispatch |
| `tools/devin_api_tool.py` | Custom CrewAI tools wrapping Devin API v3 |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

Required variables:

| Variable | Description |
|---|---|
| `DEVIN_API_KEY` | Your Devin API key (`cog_…`) |
| `DEVIN_ORG_ID` | Your Devin organization ID (`org-…`) |
| `DEVIN_USER_ID` | *(optional)* User ID to attribute sessions to |
| `OPENAI_API_KEY` | OpenAI key for the LLM backing CrewAI agents |

### 3. Run

**CLI mode** (pass request as argument):

```bash
python main.py "เพิ่มปุ่ม dark mode ในหน้า settings"
```

**Interactive mode** (prompts for input):

```bash
python main.py
```

## How It Works

1. You type a simple request in **Thai**.
2. The **Project Manager Agent** (CrewAI):
   - Translates it to English.
   - Expands it into a detailed technical spec with acceptance criteria.
3. The agent calls the **`devin_create_session`** tool which hits `POST /v3/organizations/{org_id}/sessions`.
4. A new Devin session starts working on the task automatically.
5. You get back a Thai summary + the Devin session URL.

## Devin API Tools

### `devin_create_session`
Creates a new Devin session. Accepts a detailed English prompt and returns the session URL.

### `devin_get_session`
Retrieves the status and details of an existing Devin session by ID.

## License

MIT
