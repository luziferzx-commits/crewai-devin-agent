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
            f"1. **Translate** the Thai request into professional, "
            f"software-engineering-grade English.\n"
            f"2. **Classify & Enrich** — identify the application type "
            f"(Web, Mobile, IoT, Desktop, etc.) and analyse which features "
            f"the user may have forgotten to mention (Feature Enrichment). "
            f"List the enriched features clearly.\n"
            f"3. **Risk Analysis** — summarise Security risks and Performance "
            f"risks/concerns for this system as bullet-point headings.\n"
            f"4. **Compile** everything into a structured Technical Requirements "
            f"document containing:\n"
            f"   - Project summary (1–2 sentences)\n"
            f"   - Application type classification\n"
            f"   - User stories (3–5 key stories in 'As a … I want … so that …' format)\n"
            f"   - Enriched features (features the user didn't mention but should have)\n"
            f"   - Security risks (bullet points)\n"
            f"   - Performance risks (bullet points)\n"
            f"   - Suggested technology stack\n"
            f"   - Acceptance criteria for each story\n"
        ),
        expected_output=(
            "A structured Technical Requirements Document in English with: "
            "app type classification, user stories, enriched features, "
            "security/performance risk analysis, and technology recommendations."
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
            "2. **Format** it into a single, self-contained prompt for Devin AI.\n"
            "   The prompt must include ALL technical details, stack choices,\n"
            "   implementation steps, and acceptance criteria.\n"
            "   Devin will NOT see any previous context — the prompt must stand alone.\n"
            "3. Output ONLY the formatted Devin prompt in English — no commentary,\n"
            "   no wrapper text, no Thai summary. Just the prompt itself.\n"
        ),
        expected_output=(
            "A single, self-contained English prompt for Devin AI containing "
            "all technical details, implementation steps, and acceptance criteria."
        ),
        agent=agent,
        context=[architecture_task],
    )
