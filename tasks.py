"""Task definitions for the 3-agent CrewAI → Devin pipeline.

Pipeline: Researcher → Architect → Dispatcher (sequential)
"""

from __future__ import annotations

from crewai import Agent, Task


def build_research_task(agent: Agent, thai_request: str) -> Task:
    """Task 1 — Research & Requirements (Gemini).

    Translate the Thai request, gather context, and produce raw requirements.
    """
    return Task(
        description=(
            f"## User Request (Thai)\n"
            f"{thai_request}\n\n"
            f"## Your Instructions\n"
            f"1. **Translate** the Thai request above into English.\n"
            f"2. **Research** the problem domain — identify relevant trends,\n"
            f"   best practices, and potential challenges.\n"
            f"3. **Produce** a structured Requirements Document containing:\n"
            f"   - Project summary (1–2 sentences)\n"
            f"   - User stories (3–5 key stories in 'As a … I want … so that …' format)\n"
            f"   - Acceptance criteria for each story\n"
            f"   - Suggested technology stack\n"
            f"   - Non-functional requirements (performance, responsive design, etc.)\n"
        ),
        expected_output=(
            "A structured Requirements Document in English with user stories, "
            "acceptance criteria, and technology recommendations."
        ),
        agent=agent,
    )


def build_architecture_task(agent: Agent, research_task: Task) -> Task:
    """Task 2 — Architecture & Technical Specification (Claude).

    Design the system architecture and write a detailed technical spec.
    """
    return Task(
        description=(
            "## Your Instructions\n"
            "Based on the Requirements Document from the Researcher, produce a\n"
            "**detailed Technical Specification** that includes:\n\n"
            "1. **System Architecture** — high-level component diagram description\n"
            "2. **Database Schema** — tables/collections, relationships, key fields\n"
            "3. **API Design** — endpoints, methods, request/response shapes\n"
            "4. **Frontend Structure** — pages, components, routing\n"
            "5. **Technology Stack** — finalise choices with versions\n"
            "6. **Implementation Plan** — ordered steps a developer should follow\n"
            "7. **Acceptance Criteria** — how to verify the implementation is correct\n\n"
            "Write the spec so it is precise enough for an AI coding agent (Devin)\n"
            "to implement without further clarification. Keep it under 800 words.\n"
        ),
        expected_output=(
            "A detailed Technical Specification document with architecture, "
            "DB schema, API design, frontend structure, and implementation plan."
        ),
        agent=agent,
        context=[research_task],
    )


def build_dispatch_task(agent: Agent, architecture_task: Task) -> Task:
    """Task 3 — Dispatch to Devin (Claude + Devin Tools).

    Format the spec as a Devin prompt and create a new session.
    """
    return Task(
        description=(
            "## Your Instructions\n"
            "1. Take the Technical Specification from the Architect.\n"
            "2. **Format** it into an optimal prompt for Devin — include all\n"
            "   technical details, stack choices, and acceptance criteria.\n"
            "   The prompt should be self-contained (Devin won't see previous context).\n"
            "3. **Call** the `devin_create_session` tool with the formatted prompt.\n"
            "4. **Return** a summary in Thai that includes:\n"
            "   - What was dispatched (brief project description)\n"
            "   - Key technical decisions made\n"
            "   - The Devin session URL for tracking\n"
        ),
        expected_output=(
            "A Thai-language summary confirming the task was dispatched to Devin, "
            "including key technical decisions and the Devin session URL."
        ),
        agent=agent,
        context=[architecture_task],
    )
