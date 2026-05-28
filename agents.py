"""Agent definitions for the CrewAI → Devin multi-agent system.

Three-agent pipeline:
  1. Researcher (Gemini 3.5 Flash) — gather info, summarise trends & requirements
  2. Software Architect (Claude 3.5 Sonnet) — design DB schema, logic, write tech spec
  3. Dispatcher (Claude 3.5 Sonnet + Devin Tools) — dispatch the final task to Devin
"""

from __future__ import annotations

from crewai import Agent, LLM


# ─── LLM Configurations ──────────────────────────────────────────────────────
GEMINI_MODEL = "gemini/gemini-2.5-flash"
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
            "You are the front-line Agent Researcher. When you receive a Thai "
            "request from the user, you must:\n"
            "1. Translate it into professional, software-engineering-grade English.\n"
            "2. Classify the application type (Web, Mobile, IoT, etc.) and perform "
            "Feature Enrichment — identify features the user may have forgotten.\n"
            "3. Summarise Security and Performance risks as bullet points.\n"
            "4. Compile everything into a structured Technical Requirements "
            "document to hand off to the Software Architect."
        ),
        backstory=(
            "You are a senior bilingual (Thai ↔ English) research analyst with "
            "deep expertise in software engineering. You excel at taking vague "
            "requests and enriching them — classifying app types, discovering "
            "missing features, flagging security/performance risks, and producing "
            "comprehensive Technical Requirements documents."
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
    """Agent 3 — Dispatcher (Claude 3.5 Sonnet).

    Takes the Technical Specification and formats it as an optimal
    Devin prompt. The actual API call is handled by main.py.
    """
    return Agent(
        role="Devin Dispatcher",
        goal=(
            "Take the final Technical Specification and format it as an "
            "optimal, self-contained prompt for Devin AI. The prompt must "
            "include all technical details, stack choices, implementation "
            "steps, and acceptance criteria. Output ONLY the Devin prompt "
            "in English — nothing else."
        ),
        backstory=(
            "You are an expert at writing prompts for Devin AI. You know "
            "how to structure tasks for maximum effectiveness — clear sections, "
            "explicit acceptance criteria, and zero ambiguity. You produce a "
            "single, self-contained prompt that Devin can execute independently."
        ),
        llm=_claude_llm(),
        verbose=True,
        allow_delegation=False,
    )
