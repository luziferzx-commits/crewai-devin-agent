"""Agent definitions for the CrewAI → Devin multi-agent system."""

from __future__ import annotations

from crewai import Agent

from tools.devin_api_tool import DevinCreateSessionTool, DevinGetSessionTool


def build_project_manager_agent() -> Agent:
    """Build a Project Manager Agent.

    Responsibilities:
    1. Receive a simple request in Thai.
    2. Translate and expand it into a detailed, English technical task.
    3. Call the Devin API to create a new session with that task.
    """
    return Agent(
        role="Project Manager",
        goal=(
            "Translate simple Thai-language user requests into detailed, "
            "well-structured English technical task specifications, then "
            "trigger a new Devin session to execute each task."
        ),
        backstory=(
            "You are a senior bilingual (Thai ↔ English) project manager "
            "who deeply understands software engineering. When a user sends "
            "a short request in Thai, you analyse it, infer all necessary "
            "technical details, and produce a comprehensive Devin-ready "
            "prompt in English. You then use the Devin API tool to dispatch "
            "the task automatically."
        ),
        tools=[DevinCreateSessionTool(), DevinGetSessionTool()],
        verbose=True,
        allow_delegation=False,
    )
