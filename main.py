#!/usr/bin/env python3
"""
CrewAI → Devin Multi-Agent Pipeline
=====================================
A 3-agent system that processes Thai requests through:
  1. Researcher (Gemini 2.5 Flash) — translate & gather requirements
  2. Software Architect (Claude Sonnet 4.6) — design & write tech spec
  3. Dispatcher (Claude Sonnet 4.6) — format optimal Devin prompt

After the crew finishes, main.py calls the Devin API directly to create
a session with the dispatcher's output.

Usage:
    python main.py "เพิ่มระบบ login ด้วย LINE OA"
    python main.py                                  # interactive mode
"""

from __future__ import annotations

import os
import sys

import httpx
from crewai import Crew, Process
from dotenv import load_dotenv

from agents import (
    build_architect_agent,
    build_dispatcher_agent,
    build_researcher_agent,
)
from tasks import (
    build_architecture_task,
    build_dispatch_task,
    build_research_task,
)


def create_devin_session(prompt: str) -> dict:
    """Call the Devin API v3 to create a new session."""
    api_key = os.environ.get("DEVIN_API_KEY", "")
    org_id = os.environ.get("DEVIN_ORG_ID", "")
    user_id = os.environ.get("DEVIN_USER_ID")

    if not api_key or not org_id:
        return {"error": "DEVIN_API_KEY or DEVIN_ORG_ID not set"}

    url = f"https://api.devin.ai/v3/organizations/{org_id}/sessions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: dict = {"prompt": prompt}
    if user_id:
        body["create_as_user_id"] = user_id

    resp = httpx.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run(thai_request: str) -> str:
    """Run the 3-agent pipeline: Researcher → Architect → Dispatcher → Devin API."""

    # Build agents
    researcher = build_researcher_agent()
    architect = build_architect_agent()
    dispatcher = build_dispatcher_agent()

    # Build sequential tasks (each depends on the previous)
    task_research = build_research_task(researcher, thai_request)
    task_architecture = build_architecture_task(architect, task_research)
    task_dispatch = build_dispatch_task(dispatcher, task_architecture)

    # Assemble the crew
    crew = Crew(
        agents=[researcher, architect, dispatcher],
        tasks=[task_research, task_architecture, task_dispatch],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    devin_prompt = str(result)

    # Call Devin API directly with the dispatcher's output
    print("\n🚀 กำลังส่งงานไป Devin API...")
    try:
        data = create_devin_session(devin_prompt)
        session_id = data.get("session_id", "unknown")
        session_url = data.get("url", f"https://app.devin.ai/sessions/{session_id}")
        return (
            f"✅ ส่งงานให้ Devin เรียบร้อยแล้ว!\n\n"
            f"📋 สรุป:\n"
            f"   Session ID : {session_id}\n"
            f"   URL        : {session_url}\n\n"
            f"🔗 ติดตามความคืบหน้าได้ที่: {session_url}"
        )
    except httpx.HTTPStatusError as exc:
        return f"❌ Devin API error {exc.response.status_code}: {exc.response.text}"
    except Exception as exc:
        return f"❌ Failed to create Devin session: {exc}"


def main() -> None:
    load_dotenv()

    if len(sys.argv) > 1:
        thai_request = " ".join(sys.argv[1:])
    else:
        print("🤖 CrewAI → Devin Multi-Agent Pipeline")
        print("=" * 45)
        print("ทีม Agent 3 ตัว: Researcher (Gemini) → Architect (Claude) → Dispatcher (Claude) → Devin")
        print("พิมพ์คำสั่งเป็นภาษาไทย แล้วระบบจะทำงานอัตโนมัติ\n")
        thai_request = input("📝 คำสั่ง (Thai): ").strip()
        if not thai_request:
            print("❌ ไม่มีคำสั่ง — ออกจากโปรแกรม")
            sys.exit(1)

    print(f"\n🔄 กำลังประมวลผล: {thai_request}")
    print(f"   → Agent 1 (Gemini): Research & Requirements")
    print(f"   → Agent 2 (Claude): Architecture & Tech Spec")
    print(f"   → Agent 3 (Claude): Format Devin Prompt")
    print(f"   → Devin API: Create Session\n")

    output = run(thai_request)
    print("\n" + "=" * 45)
    print("📋 ผลลัพธ์สุดท้าย:")
    print(output)


if __name__ == "__main__":
    main()
