"""One chat question through the Jev loop, after the Jev tree handed it over.

There is nothing after the loop: it always returns the answer. Without
TYPESAFE_API_KEY, or when Jev cannot be reached, code takes the steps (the SQL
agent once, then the answer analyst); a decision that fails later in the loop
ends it, and the answer analyst writes from what was found.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Any

from .. import tracing
from ..crew.chat_crew import ChatRun
from ..crew.tools import ChatContext, build_tools
from ..jev_tree.chat import ENGINE as TREE_ENGINE
from ..jev_tree.chat import Handoff
from ..search import read_catalogs
from .agents import run_analyst, run_sql_agent, run_viz_agent
from .router import MAX_STEPS, USD_PER_INPUT_TOKEN, next_step, steps_taken
from .state import LoopState

logger = logging.getLogger(__name__)

ENGINE = "jev-loop"

_client: Any = None
_client_lock = threading.Lock()


def _jev() -> Any:
    """One client for the whole server: its connection is reused between questions."""
    global _client
    with _client_lock:
        if _client is None:
            from typesafe_sdk import TypeSafeClient

            _client = TypeSafeClient()
        return _client


def _client_or_none() -> Any:
    if not os.environ.get("TYPESAFE_API_KEY"):
        return None
    try:
        return _jev()
    except Exception as exc:          # a broken install, a bad setting: code takes the steps
        logger.warning("Jev loop: no Jev client, code takes the steps: %s", exc)
        return None


def _state(run: ChatRun, handoff: Handoff | None) -> LoopState | None:
    allowed = frozenset(run.allowed_tables) if run.allowed_tables is not None else None
    catalogs = read_catalogs(Path(run.catalog_dir))
    if allowed is not None:
        catalogs = {name: c for name, c in catalogs.items() if name in allowed}
    if not catalogs:
        return None
    ctx = ChatContext(
        db_path=Path(run.db_path), catalogs=catalogs, allowed_tables=allowed,
        charts_dir=Path(run.charts_dir), qdrant_url=run.qdrant_url, collection=run.collection,
        embed_model=run.embed_model,
        previous_query={"sql": run.previous_sql, "table": run.previous_table} if run.previous_sql else None,
    )
    state = LoopState(run=run, ctx=ctx, tree_route=handoff.route if handoff else None,
                      first_step=handoff.next_step if handoff else None)
    if handoff is not None:
        if handoff.notes:
            state.reports["tree"] = handoff.notes
        if handoff.query is not None:          # the tree found the rows: they are the state's data now
            ctx.last_query = handoff.query
            state.tree_found = True
    return state


async def answer_with_loop(run: ChatRun, handoff: Handoff | None = None) -> dict[str, Any]:
    state = _state(run, handoff)
    if state is None:
        text = ("No tables are open to this account." if run.allowed_tables is not None
                else "There are no table passports yet — nothing to answer from.")
        return {"type": "error", "engine": ENGINE, "text": text}
    sql_tools, viz_tools = build_tools(state.ctx)
    client = _client_or_none()

    with tracing.span("jev-loop", input={"question": run.question,
                                         "tree": {"reason": handoff.reason, "route": handoff.route}
                                         if handoff else None}):
        for _ in range(MAX_STEPS):
            try:
                step = await asyncio.to_thread(next_step, client, state)
            except Exception as exc:     # Jev unreachable, a limit, a key: code takes the steps
                logger.warning("Jev loop: no decision from Jev, code takes the steps: %s", exc)
                client = None
                step = next_step(None, state)
                state.steps[-1]["error"] = str(exc)[:200]
            if step == "exit":
                break
            try:
                if step == "sql":
                    state.reports["sql"] = await run_sql_agent(state, sql_tools)
                else:
                    state.reports["visualizer"] = await run_viz_agent(state, viz_tools)
            except Exception as exc:     # an API error, a timeout: the analyst says what is missing
                logger.warning("Jev loop: the %s agent failed: %s", step, exc)
                state.reports[step] = f"The {step} step failed: {exc}"
        try:
            text = await run_analyst(state)
        except Exception as exc:
            logger.warning("Jev loop: the answer analyst failed: %s", exc)
            return {"type": "error", "engine": ENGINE, "text": f"The assistant could not answer: {exc}"}
        tracing.update(output={"steps": state.steps, "chart": state.ctx.chart_id is not None})

    query = state.ctx.last_query or {}
    calls = [*(handoff.calls if handoff else []), *state.jev_calls]
    tokens = sum(call["tokens"] for call in calls)
    return {
        "type": "answer",
        "engine": TREE_ENGINE if state.tree_found and not state.runs_of("sql") else ENGINE,
        "text": text,
        "table": query.get("table"),
        "sql": query.get("sql"),
        "columns": query.get("columns", []),
        "rows": query.get("rows", []),
        "chart": f"/api/charts/{state.ctx.chart_id}" if state.ctx.chart_id else None,
        "usage": {
            "tree_handoff": handoff.reason if handoff else None, "tree_route": state.tree_route,
            "steps": steps_taken(state),
            "jev_calls": len(calls), "jev_tokens": tokens,      # the tree's calls and the loop's
            "jev_cost_usd": round(tokens * USD_PER_INPUT_TOKEN, 6),
            "searches": state.ctx.searches, "queries": state.ctx.queries, "charts": state.ctx.charts,
            "agent_tokens": state.agent_tokens,
        },
    }
