"""Custom CrewAI tools for interacting with the Devin REST API (v3)."""

from __future__ import annotations

import os
from typing import Optional

import httpx
from crewai.tools import BaseTool
from pydantic import Field


DEVIN_API_BASE = "https://api.devin.ai/v3"


def _headers() -> dict[str, str]:
    api_key = os.environ.get("DEVIN_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEVIN_API_KEY environment variable is not set")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _org_id() -> str:
    org_id = os.environ.get("DEVIN_ORG_ID", "")
    if not org_id:
        raise RuntimeError("DEVIN_ORG_ID environment variable is not set")
    return org_id


class DevinCreateSessionTool(BaseTool):
    """Create a new Devin session via the v3 REST API."""

    name: str = "devin_create_session"
    description: str = (
        "Creates a new Devin session (triggers a new job). "
        "Input MUST be a detailed technical prompt in English describing the task. "
        "Returns the session URL and ID so you can track progress."
    )
    # Optional: attribute session to a specific user
    create_as_user_id: Optional[str] = Field(
        default=None,
        description="User ID to attribute the session to (optional).",
    )

    def _run(self, prompt: str) -> str:
        org_id = _org_id()
        url = f"{DEVIN_API_BASE}/organizations/{org_id}/sessions"

        body: dict = {"prompt": prompt}

        user_id = self.create_as_user_id or os.environ.get("DEVIN_USER_ID")
        if user_id:
            body["create_as_user_id"] = user_id

        try:
            resp = httpx.post(url, headers=_headers(), json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            session_id = data.get("session_id", "unknown")
            session_url = data.get("url", f"https://app.devin.ai/sessions/{session_id}")
            return (
                f"✅ Devin session created!\n"
                f"   Session ID : {session_id}\n"
                f"   URL        : {session_url}\n"
                f"   Prompt     : {prompt[:120]}…"
            )
        except httpx.HTTPStatusError as exc:
            return f"❌ API error {exc.response.status_code}: {exc.response.text}"
        except Exception as exc:
            return f"❌ Failed to create session: {exc}"


class DevinGetSessionTool(BaseTool):
    """Retrieve details about an existing Devin session."""

    name: str = "devin_get_session"
    description: str = (
        "Get the current status and details of a Devin session. "
        "Input should be a Devin session ID (UUID string)."
    )

    def _run(self, session_id: str) -> str:
        org_id = _org_id()
        devin_id = session_id if session_id.startswith("devin-") else f"devin-{session_id}"
        url = f"{DEVIN_API_BASE}/organizations/{org_id}/sessions/{devin_id}"

        try:
            resp = httpx.get(url, headers=_headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return (
                f"Session: {data.get('session_id')}\n"
                f"Status : {data.get('status')} ({data.get('status_detail', '')})\n"
                f"Title  : {data.get('title', 'N/A')}\n"
                f"URL    : {data.get('url', 'N/A')}\n"
                f"PRs    : {data.get('pull_requests', [])}"
            )
        except httpx.HTTPStatusError as exc:
            return f"❌ API error {exc.response.status_code}: {exc.response.text}"
        except Exception as exc:
            return f"❌ Failed to get session: {exc}"
