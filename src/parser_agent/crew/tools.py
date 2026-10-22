"""The tools of the chat crew, built anew for every question.

Each question gets its own ChatContext, and its tools close over it: the
database and passports it may read, what has been found so far and how much of
its budget is left. Nothing carries over from one question to the next.

The budgets are the guard against a crew that goes round in circles: a model is
told to stop in its prompt, but a tool stops it for real. Past its budget a tool
answers "no more calls — answer with what you have".

What the user is shown besides the text — the SQL, the rows, the chart — comes
from here, from what the tools actually ran and drew, never from the words the
agents chose to repeat.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crewai.tools import tool

from .. import tracing
from . import progress
from ..chat import MAX_ROWS, passport_for_prompt, rows_for_prompt, run_query
from ..search import find_tables
from .sandbox import run_chart_code

MAX_SEARCHES = 4
MAX_QUERIES = 10
MAX_CHARTS = 3

# How many candidate tables come back with their full passport.
PASSPORTS_SHOWN = 3
# Values of one column in a passport the analyst is shown.
VALUES_SHOWN = 200

# How close a word of the question must be to a name in the data to count as
# its misspelling (one wrong letter in a seven-letter name is 0.86). Lower and a
# different name would pass for a typo — the very substitution the agents must
# not make.
TYPO_CUTOFF = 0.8
_WORD = re.compile(r"[^\W\d_]{4,}", re.UNICODE)


def vocabulary(catalogs: dict[str, dict[str, Any]]) -> set[str]:
    """The words the data spells: parts of column names and of the text values."""
    words: set[str] = set()
    for catalog in catalogs.values():
        for column in catalog.get("columns") or []:
            words.update(part for part in str(column.get("column_name", "")).lower().split("_") if len(part) >= 4)
            for value in column.get("allowed_data") or []:
                words.update(w.lower() for w in _WORD.findall(str(value)))
    return words


def misspellings(question: str, catalogs: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    """(word of the question, the name in the data it most likely misspells).

    Only near spellings count; a word with no close name in the data is left
    alone — it is something the data does not hold, not a typo.
    """
    vocab = vocabulary(catalogs)
    found: list[tuple[str, str]] = []
    for word in dict.fromkeys(w.lower() for w in _WORD.findall(question)):
        # The word as written and without a grammatical ending of up to three
        # letters: a name with an Azerbaijani genitive "-in" or a plural "-s".
        if word in vocab:
            continue
        forms = [word] + [word[:-cut] for cut in (1, 2, 3) if len(word) - cut >= 4]
        exact = next((form for form in forms[1:] if form in vocab), None)
        if exact:
            found.append((word, exact))   # an ending or one letter too many
            continue
        for form in forms:
            close = difflib.get_close_matches(form, vocab, n=1, cutoff=TYPO_CUTOFF)
            if close:
                found.append((word, close[0]))
                break
    return found


@dataclass
class ChatContext:
    """One question: what its crew may read, and what it has found so far."""

    db_path: Path
    catalogs: dict[str, dict[str, Any]]
    charts_dir: Path
    qdrant_url: str
    collection: str
    embed_model: str
    # The query behind the previous answer of this conversation: a follow-up
    # such as "draw it" is charted from its rows, re-read from the database.
    previous_query: dict[str, Any] | None = None   # {"sql", "table"}
    # None — the user reads every table; otherwise only these, enforced by the
    # database itself on every query (chat.table_guard).
    allowed_tables: frozenset[str] | None = None

    searches: int = 0
    queries: int = 0
    charts: int = 0
    last_query: dict[str, Any] | None = None   # {"sql", "columns", "rows", "table"}
    chart_id: str | None = None
    tables_seen: list[str] = field(default_factory=list)


def _out_of_budget(what: str) -> str:
    return (
        f"No more {what} for this question: the budget is spent. "
        "Answer with what has been found so far, or say what is missing."
    )


def _tables_in(sql: str, catalogs: dict[str, dict[str, Any]]) -> list[str]:
    """The known tables a query names — for the table chip under the answer."""
    return [name for name in catalogs if re.search(rf'"?\b{re.escape(name)}\b"?', sql)]


def _reload_previous(ctx: ChatContext) -> None:
    """The rows of the previous answer, read again — from the database, not the text."""
    sql = ctx.previous_query["sql"]
    with tracing.span("sql_query (previous answer)", input=sql):
        result = run_query(ctx.db_path, sql, allowed_tables=ctx.allowed_tables)
        tracing.update(
            output={"rows": len(result["rows"]), "error": result["error"]},
            level="ERROR" if result["error"] else None,
        )
    if not result["error"]:
        ctx.last_query = {
            "sql": sql, "columns": result["columns"], "rows": result["rows"][:MAX_ROWS],
            "table": ctx.previous_query.get("table"),
        }


def build_tools(ctx: ChatContext) -> tuple[list[Any], list[Any]]:
    """(the analyst's tools, the visualization specialist's tools) for one question."""

    @tool("catalog_search")
    def catalog_search(question: str) -> str:
        """Finds the tables that hold what the question is about and returns the
        passport of each candidate: every column with its SQL type, a sample and
        the values that occur. Call it before writing any SQL, with the user's
        question in their own words."""
        if ctx.searches >= MAX_SEARCHES:
            return _out_of_budget("catalog searches")
        ctx.searches += 1
        with tracing.span("catalog_search", input=question):
            found = find_tables(
                question, ctx.catalogs, ctx.db_path,
                qdrant_url=ctx.qdrant_url, collection=ctx.collection,
                embed_model=ctx.embed_model,
            )
            tracing.update(output={
                "by": found.by, "ambiguous": found.ambiguous,
                "candidates": [c["table_name"] for c in found.candidates[:5]],
            })

        lines: list[str] = []
        typos = misspellings(question, ctx.catalogs)
        if typos:
            lines.append(
                "Likely misspellings in the question (use the spelling from the data, "
                "and say in the answer which name you took): "
                + ", ".join(f"'{word}' -> '{name}'" for word, name in typos)
            )
        if not found.candidates:
            return "No table matches this question. Say so, and suggest what could be asked."

        if not found.matched:
            lines.append(
                "Nothing in the question points at a particular table: these are all "
                "the tables there are. If none fits, say so instead of forcing one."
            )
        lines.append(f"Candidate tables, best first (ranked by {found.by}):")
        for number, item in enumerate(found.candidates[:5], start=1):
            why = f" — found: {item['why']}" if item.get("why") and isinstance(item["why"], str) else ""
            lines.append(f"{number}. {item['table_name']}: {item.get('table_description', '')}{why}")
        if found.ambiguous:
            lines.append(
                "The leading candidates are too close to call. If the question does "
                "not settle which table is meant, do not guess: report the candidates "
                "so the coordinator can ask the user."
            )

        for name in [c["table_name"] for c in found.candidates][:PASSPORTS_SHOWN]:
            catalog = ctx.catalogs.get(name)
            if catalog is not None:
                ctx.tables_seen.append(name)
                lines.append("\n" + passport_for_prompt(catalog, max_values=VALUES_SHOWN))
        return "\n".join(lines)

    @tool("sql_query")
    def sql_query(sql: str) -> str:
        """Runs ONE read-only SQLite SELECT and returns the rows (at most 200).
        Quote every identifier with double quotes and use the values from the
        passport literally. An error comes back as text: fix the query and call
        again."""
        if ctx.queries >= MAX_QUERIES:
            return _out_of_budget("SQL queries")
        ctx.queries += 1
        with tracing.span("sql_query", input=sql):
            result = run_query(ctx.db_path, sql, allowed_tables=ctx.allowed_tables)
            tracing.update(
                output={"rows": len(result["rows"]), "error": result["error"]},
                level="ERROR" if result["error"] else None,
            )
        if result["error"]:
            return f"ERROR: {result['error']}"
        tables = _tables_in(sql, ctx.catalogs)
        ctx.last_query = {
            "sql": sql, "columns": result["columns"], "rows": result["rows"][:MAX_ROWS],
            "table": tables[0] if tables else None,
        }
        return f"{len(result['rows'])} rows\n{rows_for_prompt(result)}"

    @tool("run_chart_code")
    def run_chart_code_tool(code: str) -> str:
        """Draws ONE chart with matplotlib in an isolated sandbox (no network, no
        files). A pandas DataFrame `df` already holds the rows of the last
        successful sql_query of this question — or, for a follow-up such as
        "draw it", the rows of the previous answer. `pd` and `plt` are imported.
        Give the chart a title and axis labels, then save it with
        plt.savefig(OUT). Returns "chart saved" or the error to fix."""
        if ctx.last_query is None and ctx.previous_query:
            _reload_previous(ctx)
        if ctx.last_query is None:
            return (
                "ERROR: there are no rows to draw: nothing has been queried in this "
                "conversation yet. Tell the coordinator that the Data Retrieval Analyst "
                "has to retrieve the rows first."
            )
        if ctx.charts >= MAX_CHARTS:
            return _out_of_budget("chart attempts")
        ctx.charts += 1
        columns, rows = ctx.last_query["columns"], ctx.last_query["rows"]
        with tracing.span("chart_sandbox", input=code):
            result = run_chart_code(code, columns, rows, ctx.charts_dir)
            tracing.update(
                output={"chart": result.chart_id, "error": result.error},
                level="ERROR" if result.error else None,
            )
        if not result.ok:
            return f"ERROR: {result.error}\n(df columns: {columns}; {len(rows)} rows)"
        ctx.chart_id = result.chart_id
        progress.chart_saved()
        return f"chart saved (df had {len(rows)} rows, columns {columns})"

    return [catalog_search, sql_query], [run_chart_code_tool]
