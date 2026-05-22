#!/usr/bin/env python3
"""
CrewAI → Devin Multi-Agent Pipeline
=====================================
A 3-agent system that processes Thai requests through:
  1. Researcher (Gemini 3.5 Flash) — translate & gather requirements
  2. Software Architect (Claude 3.5 Sonnet) — design & write tech spec
  3. Dispatcher (Claude 3.5 Sonnet + Devin Tools) — dispatch to Devin

Usage:
    python main.py "เพิ่มระบบ login ด้วย LINE OA"
    python main.py                                  # interactive mode
"""

from __future__ import annotations

import sys

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


def run(thai_request: str) -> str:
    """Run the 3-agent pipeline: Researcher → Architect → Dispatcher."""

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
    return str(result)


def main() -> None:
    load_dotenv()

    if len(sys.argv) > 1:
        thai_request = " ".join(sys.argv[1:])
    else:
        print("🤖 CrewAI → Devin Multi-Agent Pipeline")
        print("=" * 45)
        print("ทีม Agent 3 ตัว: Researcher (Gemini) → Architect (Claude) → Dispatcher (Claude + Devin)")
        print("พิมพ์คำสั่งเป็นภาษาไทย แล้วระบบจะทำงานอัตโนมัติ\n")
        thai_request = input("📝 คำสั่ง (Thai): ").strip()
        if not thai_request:
            print("❌ ไม่มีคำสั่ง — ออกจากโปรแกรม")
            sys.exit(1)

    print(f"\n🔄 กำลังประมวลผล: {thai_request}")
    print(f"   → Agent 1 (Gemini): Research & Requirements")
    print(f"   → Agent 2 (Claude): Architecture & Tech Spec")
    print(f"   → Agent 3 (Claude): Dispatch to Devin\n")

    output = run(thai_request)
    print("\n" + "=" * 45)
    print("📋 ผลลัพธ์สุดท้าย:")
    print(output)


if __name__ == "__main__":
    main()
