"""Agent definitions for the CrewAI → Devin multi-agent system.

Three-agent pipeline:
  1. Researcher (Gemini 3.5 Flash) — gather info, summarise trends & requirements
  2. Software Architect (Claude 3.5 Sonnet) — design DB schema, logic, write tech spec
  3. Dispatcher (Claude 3.5 Sonnet + Devin Tools) — dispatch the final task to Devin
"""

from __future__ import annotations

from crewai import Agent, LLM

from tools.devin_api_tool import DevinCreateSessionTool, DevinGetSessionTool

# ─── LLM Configurations ──────────────────────────────────────────────────────
GEMINI_MODEL = "gemini/gemini-3.5-flash"
CLAUDE_MODEL = "anthropic/claude-sonnet-4-6"


def _gemini_llm() -> LLM:
    return LLM(model=GEMINI_MODEL)


def _claude_llm() -> LLM:
    return LLM(model=CLAUDE_MODEL, temperature=0.2)


# ─── Agent Builders ───────────────────────────────────────────────────────────

def build_researcher_agent() -> Agent:
    """Agent 1 — Researcher (Gemini 3.5 Flash).

    Translates the Thai request, researches context, summarises trends,
    and produces raw requirements.
    """
    return Agent(
        role="Researcher",
        goal=(
            "Translate the user's Thai request into English, research relevant "
            "context, summarise industry trends, and produce a clear set of "
            "raw requirements and user stories for the development team."
        ),
        backstory=(
            "You are a senior bilingual (Thai ↔ English) research analyst with "
            "deep knowledge of software engineering trends. You excel at taking "
            "vague requests and turning them into well-researched requirement "
            "documents with clear user stories and acceptance criteria."
        ),
        llm=_gemini_llm(),
        verbose=True,
        allow_delegation=False,
    )


def build_architect_agent() -> Agent:
    """Agent 2 — Software Architect (Claude 3.5 Sonnet).

    Designs system architecture, DB schema, and writes a detailed
    Technical Specification based on the Researcher's output.
    """
    return Agent(
        role="Software Architect",
        goal=(
            "Take the raw requirements and produce a detailed Technical "
            "Specification including: system architecture, database schema, "
            "API endpoints, technology stack recommendations, and a step-by-step "
            "implementation plan that Devin can follow."
        ),
        backstory=(
            "You are a principal software architect with 15+ years of experience "
            "designing scalable systems. You think in terms of clean architecture, "
            "separation of concerns, and developer experience. You produce specs "
            "that are precise enough for an AI coding agent (Devin) to implement "
            "without ambiguity."
        ),
        llm=_claude_llm(),
        verbose=True,
        allow_delegation=False,
    )


def build_dispatcher_agent() -> Agent:
    """Agent 3 — Dispatcher (Claude 3.5 Sonnet + Devin Tools).

    Takes the Technical Specification and dispatches it as a new Devin
    session via the API.
    """
    return Agent(
        role="Devin Dispatcher",
        goal=(
            "Take the final Technical Specification, format it as an optimal "
            "Devin prompt, and create a new Devin session to execute the task. "
            "Return the session URL and a Thai-language summary to the user."
        ),
        backstory=(
            "You are an expert at interfacing with Devin AI. You know how to "
            "write prompts that maximise Devin's effectiveness — clear, "
            "structured, with explicit acceptance criteria. After dispatching, "
            "you confirm success and provide a summary in Thai."
        ),
        llm=_claude_llm(),
        tools=[DevinCreateSessionTool(), DevinGetSessionTool()],
        verbose=True,
        allow_delegation=False,
    )
