"""Two-layer research for blocker resolution.

Layer 1 — Inline due diligence: fast parallel web + knowledge search
    with LLM triage. Completes in seconds.

Layer 2 — Deep research: full LLM session with tools for thorough
    investigation. May take minutes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from genesis_task_executor.prompts import (
    DUE_DILIGENCE_TRIAGE,
    RESEARCH_TEMPLATE,
    render_template,
)
from genesis_task_executor.protocols import (
    KnowledgeStore,
    LLMProvider,
    NoOpKnowledgeStore,
    NoOpWebSearcher,
    WebSearcher,
)
from genesis_task_executor.types import ResearchResult

logger = logging.getLogger(__name__)


class Researcher:
    """Two-layer research for step failure investigation."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        web_searcher: WebSearcher | None = None,
        knowledge_store: KnowledgeStore | None = None,
    ) -> None:
        self._provider = provider
        self._web = web_searcher or NoOpWebSearcher()
        self._knowledge = knowledge_store or NoOpKnowledgeStore()

    async def inline_due_diligence(
        self,
        step: dict,
        error: str,
    ) -> str | None:
        """Layer 1: quick parallel web + knowledge search with triage.

        Returns context string if relevant results found, None otherwise.
        """
        query = f"{step.get('description', '')[:100]} {error[-150:]}"

        # Parallel search
        web_task = self._web.search(query, max_results=5)
        knowledge_task = self._knowledge.recall(query, limit=5)

        try:
            web_results, knowledge_results = await asyncio.gather(
                web_task, knowledge_task, return_exceptions=True,
            )
        except Exception:
            logger.warning("Due diligence search failed", exc_info=True)
            return None

        # Handle exceptions from gather
        if isinstance(web_results, Exception):
            web_results = []
        if isinstance(knowledge_results, Exception):
            knowledge_results = []

        if not web_results and not knowledge_results:
            return None

        # Format results for triage
        parts: list[str] = []
        for r in web_results:
            parts.append(f"[Web] {r.get('title', '')}: {r.get('snippet', '')}")
        for r in knowledge_results:
            parts.append(f"[Knowledge] {r.get('content', '')[:300]}")
        combined = "\n\n".join(parts)

        # LLM triage: are these results relevant?
        triage_prompt = render_template(
            DUE_DILIGENCE_TRIAGE,
            step_description=step.get("description", ""),
            error_text=error,
        )

        try:
            response = await self._provider.complete([
                {"role": "system", "content": triage_prompt},
                {"role": "user", "content": combined},
            ])
        except Exception:
            logger.warning("Due diligence triage failed", exc_info=True)
            return None

        if "NOT_RELEVANT" in response:
            return None

        return response

    async def research(
        self,
        step: dict,
        error: str,
        prior_attempts: list[str],
        *,
        due_diligence_results: str | None = None,
    ) -> ResearchResult | None:
        """Layer 2: deep research session with full LLM investigation."""
        prompt = render_template(
            RESEARCH_TEMPLATE,
            step_description=step.get("description", ""),
            error_text=error,
            prior_attempts="\n".join(f"- {a}" for a in prior_attempts) or "(none)",
            due_diligence_results=due_diligence_results or "(none)",
        )

        try:
            response = await self._provider.complete([
                {"role": "user", "content": prompt},
            ])
        except Exception:
            logger.warning("Research session failed", exc_info=True)
            return None

        return _parse_research_output(response)


def _parse_research_output(text: str) -> ResearchResult:
    """Extract ResearchResult from LLM response text."""
    # Try to find JSON block
    json_match = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return ResearchResult(
                found=data.get("found", False),
                approach=data.get("approach"),
                sources=data.get("sources", []),
                clues=data.get("clues"),
                concrete_blockers=data.get("concrete_blockers", []),
            )
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: try to parse the entire response as JSON
    try:
        data = json.loads(text)
        return ResearchResult(
            found=data.get("found", False),
            approach=data.get("approach"),
            sources=data.get("sources", []),
            clues=data.get("clues"),
            concrete_blockers=data.get("concrete_blockers", []),
        )
    except (json.JSONDecodeError, KeyError):
        pass

    # Last resort: treat entire response as clue text
    return ResearchResult(found=False, clues=text[:500])
