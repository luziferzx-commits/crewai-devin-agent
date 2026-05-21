#!/usr/bin/env python3
"""
CrewAI → Devin Project Manager
===============================
A multi-agent system that translates simple Thai requests into
detailed technical tasks and triggers Devin sessions automatically.

Usage:
    python main.py "เพิ่มปุ่ม dark mode ในหน้า settings"
    python main.py                           # interactive mode
"""

from __future__ import annotations

import sys

from crewai import Crew, Process
from dotenv import load_dotenv

from agents import build_project_manager_agent
from tasks import build_translate_and_dispatch_task


def run(thai_request: str) -> str:
    """Translate *thai_request* and dispatch a Devin session."""
    pm_agent = build_project_manager_agent()
    task = build_translate_and_dispatch_task(pm_agent, thai_request)

    crew = Crew(
        agents=[pm_agent],
        tasks=[task],
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
        print("🤖 CrewAI → Devin Project Manager")
        print("=" * 40)
        print("พิมพ์คำสั่งเป็นภาษาไทย แล้ว Agent จะแปลงเป็น task ให้ Devin ทำงานอัตโนมัติ\n")
        thai_request = input("📝 คำสั่ง (Thai): ").strip()
        if not thai_request:
            print("❌ ไม่มีคำสั่ง — ออกจากโปรแกรม")
            sys.exit(1)

    print(f"\n🔄 กำลังประมวลผล: {thai_request}\n")
    output = run(thai_request)
    print("\n" + "=" * 40)
    print("📋 ผลลัพธ์:")
    print(output)


if __name__ == "__main__":
    main()
