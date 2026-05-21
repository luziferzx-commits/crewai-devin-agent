"""Task definitions for the CrewAI → Devin pipeline."""

from __future__ import annotations

from crewai import Agent, Task


def build_translate_and_dispatch_task(
    agent: Agent,
    thai_request: str,
) -> Task:
    """Build a task that translates Thai input and dispatches to Devin.

    The agent will:
    1. Translate the Thai request to English.
    2. Expand it into a detailed technical prompt with acceptance criteria.
    3. Call `devin_create_session` with the refined prompt.
    """
    return Task(
        description=(
            f"## User Request (Thai)\n"
            f"{thai_request}\n\n"
            f"## Your Instructions\n"
            f"1. **Translate** the Thai request above into English.\n"
            f"2. **Expand** it into a detailed technical task specification:\n"
            f"   - Describe what needs to be built or changed.\n"
            f"   - List acceptance criteria / expected behaviour.\n"
            f"   - Mention relevant technologies if you can infer them.\n"
            f"   - Keep the prompt concise but complete (≤ 500 words).\n"
            f"3. **Dispatch** by calling the `devin_create_session` tool with\n"
            f"   the refined English prompt.\n"
            f"4. Return a short summary in Thai confirming what was dispatched,\n"
            f"   including the Devin session URL.\n"
        ),
        expected_output=(
            "A Thai-language summary confirming the task was dispatched, "
            "including the Devin session URL."
        ),
        agent=agent,
    )
