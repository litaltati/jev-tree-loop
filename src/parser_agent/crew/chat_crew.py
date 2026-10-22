"""One chat question through the crew: built, run, and read back.

    question ─▶ Data Assistant Coordinator (manager of a hierarchical crew)
                  ├─▶ Data Retrieval Analyst: catalog_search → sql_query
                  └─▶ Data Visualization Specialist: run_chart_code (when a chart helps)
             ◀─ the answer text, plus the SQL, rows and chart the tools produced

A new crew, with new tools and a new context, is built for every question.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from crewai import Crew, Process, Task
from pydantic import BaseModel, Field

from ..search import read_catalogs
from .agents import build_agents
from .tools import ChatContext, build_tools

logger = logging.getLogger(__name__)

# How many earlier turns of the conversation the coordinator is shown.
MAX_HISTORY_TURNS = 6
MAX_TURN_CHARS = 2000


class ChatRun(BaseModel):
    """One question for the chat process of the flow, and its answer."""

    question: str
    history: list[dict[str, str]] = Field(default_factory=list)
    previous_sql: str | None = None          # the query behind the previous answer
    previous_table: str | None = None
    db_path: str
    catalog_dir: str
    charts_dir: str
    qdrant_url: str
    collection: str
    embed_model: str
    models: dict[str, str]                   # orchestrator / sql / viz
    allowed_tables: list[str] | None = None  # None — the user reads every table
    answer: dict[str, Any] | None = None


def task_description(run: ChatRun) -> str:
    lines = [f'A user of the data catalog asks: "{run.question}"']
    turns = [turn for turn in run.history if turn.get("content")][-MAX_HISTORY_TURNS:]
    if turns:
        lines.append("\nEarlier in this conversation:")
        for turn in turns:
            who = "Assistant" if turn.get("role") == "assistant" else "User"
            lines.append(f"- {who}: {str(turn['content'])[:MAX_TURN_CHARS]}")
    if run.previous_sql:
        lines.append(
            f"\nThe previous answer came from this query on the table "
            f"{run.previous_table or '(unknown)'}:\n{run.previous_sql}\n"
            "Its rows are loaded for the Data Visualization Specialist as they are: "
            "a chart of the previous answer needs no new query."
        )
    lines.append(
        "\nAnswer the question from the database. Delegate finding the table and "
        "running the SQL to the Data Retrieval Analyst. When the user wants a chart, "
        "or a chart clearly helps, delegate it to the Data Visualization Specialist — "
        "straight away for a chart of the previous answer, after the analyst for new data."
    )
    return "\n".join(lines)


EXPECTED_OUTPUT = (
    "A short, specific answer for the user in English: the figures, what they are "
    "and which table they came from. Do not copy the query rows into the text: the "
    "page shows them as a table under the answer. A Markdown table only for a small "
    "comparison (at most 10 rows and 6 columns), in English, with the names as they "
    "appear in the data. If a name was misspelled, say which name "
    "from the data you used. If only part of what was asked is in the data, give "
    "that part and say for each missing item why it is missing — never substitute "
    "something else for it. Mention a chart only if the Data Visualization "
    "Specialist reported \"chart saved\"; if it could not draw one, say so. If the "
    "question cannot be answered from the data, say why; if it is unclear which "
    "table is meant, ask one clarifying question."
)


async def answer_question(run: ChatRun, *, verbose: bool = False) -> dict[str, Any]:
    """Runs the crew for one question. The result is what the web page renders."""
    allowed = frozenset(run.allowed_tables) if run.allowed_tables is not None else None
    catalogs = read_catalogs(Path(run.catalog_dir))
    if allowed is not None:
        # A table the user may not read does not exist for this crew: not in the
        # search, not in a passport. The database refuses it as well.
        catalogs = {name: passport for name, passport in catalogs.items() if name in allowed}
    ctx = ChatContext(
        db_path=Path(run.db_path),
        catalogs=catalogs,
        allowed_tables=allowed,
        charts_dir=Path(run.charts_dir),
        qdrant_url=run.qdrant_url,
        collection=run.collection,
        embed_model=run.embed_model,
        previous_query=(
            {"sql": run.previous_sql, "table": run.previous_table} if run.previous_sql else None
        ),
    )
    if not ctx.catalogs:
        text = ("No tables are open to this account." if allowed is not None
                else "There are no table passports yet — nothing to answer from.")
        return {"type": "error", "text": text}

    sql_tools, viz_tools = build_tools(ctx)
    orchestrator, sql_agent, viz_agent = build_agents(
        run.models, sql_tools, viz_tools, verbose=verbose
    )
    task = Task(description=task_description(run), expected_output=EXPECTED_OUTPUT)
    crew = Crew(
        agents=[sql_agent, viz_agent],
        tasks=[task],
        process=Process.hierarchical,
        manager_agent=orchestrator,
        verbose=verbose,
    )

    try:
        # No inputs: the question goes into the task as it is, and a brace in it
        # must not be taken for a template placeholder.
        output = await crew.kickoff_async()
    except Exception as exc:  # a timeout, an API error, a crew that gave up
        logger.warning("crew failed: %s", exc)
        return {"type": "error", "text": f"The assistant could not answer: {exc}"}

    query = ctx.last_query or {}
    return {
        "type": "answer",
        "engine": "crew",
        "text": (output.raw or "").strip(),
        "table": query.get("table"),
        "sql": query.get("sql"),
        "columns": query.get("columns", []),
        "rows": query.get("rows", []),
        "chart": f"/api/charts/{ctx.chart_id}" if ctx.chart_id else None,
        "usage": {
            "searches": ctx.searches, "queries": ctx.queries, "charts": ctx.charts,
            "total_tokens": getattr(output.token_usage, "total_tokens", None),
        },
    }
