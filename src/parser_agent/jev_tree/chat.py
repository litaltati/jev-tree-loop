"""The Jev tree as the first step of every question: it finds the figures of one line.

    question ─▶ Jev tree (tree.py) ─┬─ one line it is sure of ─▶ SQL ─▶ the rows and notes on them
                                    ├─ more than one line, a ranking, a calculation, a table name
                                    │                         ─▶ the SQL agent goes first
                                    ├─ a company or period the data lacks ─▶ the analyst says so
                                    ├─ a definition, small talk ─▶ the analyst answers, no data
                                    └─ unsure which line, an unsure route, anything else ─▶ Jev decides

Whatever the tree finds goes into one Handoff, and the Jev loop (jev_loop/) goes
on from it with the same state: Jev picks the visualizer when a chart is wanted,
and the answer analyst always writes the answer. No LLM is called here, and the
tree writes no answer itself: its notes — the figures as the rows hold them, a
company it left out — are for the analyst. A line the tree could not find goes
to the loop rather than being reported missing: its SQL agent searches more
widely.

PARSER_AGENT_JEV_TREE=off in .env turns the tree off; without TYPESAFE_API_KEY
it is off as well.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import tracing
from ..chat import run_query
from ..crew import progress
from ..search import read_catalogs
from .names import sound
from .options import TableOptions, tree_options
from .tree import COMPANY, ROUTE_GATE, Plan, plan_question

logger = logging.getLogger(__name__)

ENABLED_ENV = "PARSER_AGENT_JEV_TREE"
ENGINE = "jev-tree"       # an answer whose figures the tree found
TABLE_NAME_CLOSE = 0.85   # a word this close to a table's name is that table, not a company

_client: Any = None
_client_lock = threading.Lock()


def tree_enabled() -> bool:
    if os.environ.get(ENABLED_ENV, "on").strip().lower() in ("off", "0", "false", "no"):
        return False
    return bool(os.environ.get("TYPESAFE_API_KEY"))


def _jev() -> Any:
    """One client for the whole server: its connection is reused between questions."""
    global _client
    with _client_lock:
        if _client is None:
            from typesafe_sdk import TypeSafeClient

            _client = TypeSafeClient()
        return _client


@dataclass
class Handoff:
    """What the tree found out, for the Jev loop that goes on from it.

    reason     why the tree stopped where it did
    route      the tree's route, only when it was sure of it
    next_step  the loop's first step when the tree already knows it: "sql" (more
               than one line, a ranking, a calculation, a table name) or "exit"
               (no data needed, or a company or period the data lacks); None — Jev decides
    query      the rows the tree found: sql, table, columns, rows
    notes      what the tree has to tell the answer analyst
    calls      the tree's Jev calls, for the usage
    """

    reason: str
    route: str | None = None
    next_step: str | None = None
    query: dict[str, Any] | None = None
    notes: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)


def _handoff(plan: Plan, reason: str, *, next_step: str | None = None, notes: str | None = None) -> Handoff:
    route = plan.route if plan.route and plan.route_confidence >= ROUTE_GATE else None
    return Handoff(reason=reason, route=route, next_step=next_step, notes=notes, calls=plan.calls)


def answer_with_tree(run: Any) -> Handoff:
    """What the tree found for this question; the Jev loop always goes on from it."""
    if not tree_enabled():
        return Handoff("the tree is off")
    allowed = frozenset(run.allowed_tables) if run.allowed_tables is not None else None
    catalogs = read_catalogs(Path(run.catalog_dir))
    if allowed is not None:
        catalogs = {name: c for name, c in catalogs.items() if name in allowed}
    options = tree_options(catalogs, Path(run.db_path))
    if not options:
        return Handoff("no table the tree can read")

    try:
        with tracing.span("jev-tree", input={"question": run.question}):
            plan = plan_question(_jev(), run.question, run.history, options)
            tracing.update(output={key: getattr(plan, key) for key in
                                   ("outcome", "reason", "hand_to", "route", "table", "line", "entities", "sql")})
    except Exception as exc:  # Jev unreachable, a limit, a key: the loop still answers
        logger.warning("Jev tree failed, the loop answers: %s", exc)
        return Handoff(f"Jev failed: {exc}"[:200])

    if plan.outcome == "llm" or (plan.outcome == "not_in_data" and plan.line is None):
        return _handoff(plan, plan.reason or plan.outcome,
                        next_step={"sql": "sql", "answer": "exit"}.get(plan.hand_to or ""))
    if plan.outcome == "ask":
        return _handoff(plan, plan.reason, notes=(
            f"The lookup could not tell which line of the table {plan.table} is meant. "
            f"The likeliest lines: {', '.join(plan.alternatives)}."))
    if plan.outcome == "not_in_data":
        if names_a_table(run.question, catalogs):
            return _handoff(plan, "the message names a table, not a company", next_step="sql")
        return _handoff(plan, plan.reason, next_step="exit", notes=_missing_text(plan, options))

    with progress.code_step(progress.DATA):
        result = run_query(Path(run.db_path), plan.sql, allowed_tables=allowed)
    if result["error"]:
        logger.warning("Jev tree query failed, the loop goes on: %s", result["error"])
        return _handoff(plan, f"the query failed: {result['error']}"[:200])
    handoff = _handoff(plan, "the tree found the figures", notes=_text(plan, result, options[plan.table]))
    handoff.query = {"sql": plan.sql, "table": plan.table, "columns": result["columns"], "rows": result["rows"]}
    return handoff


def names_a_table(message: str, catalogs: dict[str, Any]) -> bool:
    """Whether a word of the message, or two or three of them written together,
    is spelled like a table's name ("конт 3" for cont3): then it was not a
    company the data lacks, and "not in the data" would be a wrong answer."""
    keys = [re.sub(r"[^a-z0-9]", "", sound(name)) for name in catalogs]
    words = re.findall(r"[^\W_]+", message)
    for size in (1, 2, 3):
        for i in range(len(words) - size + 1):
            key = re.sub(r"[^a-z0-9]", "", sound("".join(words[i:i + size])))
            if len(key) >= 4 and (key in keys or difflib.get_close_matches(key, keys, n=1, cutoff=TABLE_NAME_CLOSE)):
                return True
    return False


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}"
    return str(value)


def _text(plan: Plan, result: dict[str, Any], table: TableOptions) -> str:
    """The tree's notes on its rows, for the answer analyst."""
    columns, rows = result["columns"][1:], result["rows"]
    if result["columns"][:1] == [COMPANY]:
        left_out = len(plan.entities) - len(rows)
        if not rows:
            text = (f'The table {plan.table} has the line "{plan.line}", but every company asked about '
                    "has 0 or no figure for it.")
        else:
            text = f"{plan.line}, from the table {plan.table}: a row per company, the largest first."
            if left_out:
                text += (f" {left_out} of the {len(plan.entities)} companies have 0 or no figure for "
                         "this line and are left out.")
    elif not rows:
        text = f'The table {plan.table} has the line "{plan.line}", but no rows came back for it.'
    elif len(rows) == 1 and len(columns) <= 6:
        lines = [f"- {column}: {_fmt(value)}" for column, value in zip(columns, rows[0][1:])]
        text = f"{plan.line}, from the table {plan.table}:\n" + "\n".join(lines)
    elif plan.breakdown:
        text = (f"{plan.line} and the {len(rows) - 1} lines it is made of, from the table {plan.table}; "
                "the figures are in the table below.")
    else:
        text = (f"{plan.line} for {len(columns)} columns, from the table {plan.table}; "
                "the figures are in the table below.")
    if plan.duplicate_line and len(rows) > 1:
        text += f'\n\nThe line "{plan.line}" occurs {len(rows)} times in the report; each is shown.'
    if plan.unlisted_entity:
        text += "\n\nOne of the companies the user named is not in the data, so it has no figures here."
    return text


def _missing_text(plan: Plan, options: dict[str, TableOptions]) -> str:
    if "period" in plan.reason and plan.table:
        table = options[plan.table]
        return (f"The table {plan.table} has no {plan.measure or 'actual'} figures for "
                f"{plan.period or 'that period'}. It has: {', '.join(table.periods)}.")
    # The companies of the table that has the most of them — the others hold
    # report formats or a single total, not a list of companies.
    widest = max(options.values(), key=lambda t: len(t.entities))
    companies = [c.title() for c in widest.entities if c]
    shown = ", ".join(companies[:12]) + (f" and {len(companies) - 12} more" if len(companies) > 12 else "")
    return ("The company you asked about is not in the data, so there are no figures for it. "
            f"The companies in {widest.name}: {shown}.")
