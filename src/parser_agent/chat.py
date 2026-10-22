"""The SQL side of the chat: what the Data Retrieval Analyst's tools stand on.

The chat itself is the Jev tree (jev_tree/) and, for what it hands over, the
Jev loop (jev_loop/): the analyst finds the table and runs SQL, the
visualization specialist draws. What stays here is what must not be left to a
model:

  * the passport as the SQL writer sees it — names, types, the values that occur;
  * only one reading statement is let through (is_safe_select), with a LIMIT;
  * the database is opened read-only, and a misspelt column name is an error,
    not a column of identical strings.

An answer must not be able to change the data it is read from.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

from .catalog import connect_existing

logger = logging.getLogger(__name__)

# How many rows come back from one query. Enough to answer by, small enough to
# put into a prompt.
MAX_ROWS = 200


# How many rows of the result the answering model is shown.
ROWS_IN_PROMPT = 40


_LEVEL_COLUMN = re.compile(r"^(?P<label>.+)_l(?P<level>\d+)$")


def hierarchy_levels(names: list[str]) -> dict[str, tuple[str, int]]:
    """{level column: (its label column, level number)}.

    When a sheet's row labels are nested, the extractor keeps the row's own
    label in a column `X` and adds the groups above it as `X_l0`, `X_l1`, ...
    (extractor.HierarchyPlan). So a level column repeats one group name on every
    row under that group, and `X` is what tells the rows apart.
    """
    present = set(names)
    levels: dict[str, tuple[str, int]] = {}
    for name in names:
        match = _LEVEL_COLUMN.match(name)
        if match and match["label"] in present:
            levels[name] = (match["label"], int(match["level"]))
    return levels


def passport_for_prompt(catalog: dict[str, Any], max_values: int = 400) -> str:
    """The passport as the SQL model sees it: names, types, values.

    The whole domain of a column goes in, not the beginning of it: a row the
    question names may be the hundredth in the list, and a model that has not
    seen it spells it by guess — and the query finds nothing.

    The columns of a row-label hierarchy are marked, from their names: the
    descriptions written for the passport do not say which is which.
    """
    lines = [
        f'Table "{catalog["table_name"]}" — {catalog.get("table_description", "")}',
        "Columns:",
    ]
    columns = catalog.get("columns") or []
    levels = hierarchy_levels([str(c.get("column_name", "")) for c in columns])
    labels: dict[str, list[str]] = {}
    for level_column, (label, _) in levels.items():
        labels.setdefault(label, []).append(level_column)
    for column in columns:
        name = str(column.get("column_name", ""))
        line = (
            f'- "{name}" {column.get("column_data_type", "")}'
            f' — {column.get("column_description", "")}'
        )
        if name in levels:
            label, level = levels[name]
            line += (
                f' [HIERARCHY LEVEL {level} of "{label}": the group the row belongs to, the '
                f'same on every row of the group. Filter by it to get a group\'s details; '
                f'never show it as the row label]'
            )
        elif name in labels:
            above = ", ".join(f'"{c}"' for c in labels[name])
            line += (
                f" [ROW LABEL: the row's own name — show this column as the row label. "
                f"The groups above it are in {above}; a group's own row has its name here]"
            )
        values = column.get("allowed_data") or []
        if values:
            shown = ", ".join(str(v) for v in values[:max_values])
            more = f" … +{len(values) - max_values}" if len(values) > max_values else ""
            line += f" | values: {shown}{more}"
        lines.append(line)
    return "\n".join(lines)


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|replace)\b",
    re.IGNORECASE,
)


def is_safe_select(sql: str) -> bool:
    """Whether this is a single reading query.

    The database is opened read-only anyway; this is the second lock, so that a
    stray statement is refused before it reaches SQLite at all.
    """
    text = sql.strip().rstrip(";").strip()
    if not text or ";" in text:
        return False
    if not re.match(r"^(select|with)\b", text, re.IGNORECASE):
        return False
    return not _FORBIDDEN.search(text)


def with_limit(sql: str, limit: int = MAX_ROWS) -> str:
    """Adds a LIMIT if the query has none: the page must not be flooded."""
    text = sql.strip().rstrip(";").strip()
    return text if re.search(r"\blimit\b\s+\d+", text, re.IGNORECASE) else f"{text} LIMIT {limit}"


def table_guard(allowed: frozenset[str]) -> Any:
    """A SQLite authorizer that lets a statement read only the allowed tables.

    SQLite asks it about every table a statement reads while the statement is
    compiled — through joins, subqueries, CTEs and views alike — so a table
    cannot be reached by spelling the query differently. The schema table is
    closed too: a restricted user does not get to list what else exists.
    """
    allowed_lower = {name.lower() for name in allowed}

    def authorize(action: int, table: str | None, *_: Any) -> int:
        if action == sqlite3.SQLITE_READ and (table or "").lower() not in allowed_lower:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    return authorize


def run_query(
    db_path: Path, sql: str, *, allowed_tables: frozenset[str] | None = None
) -> dict[str, Any]:
    """Runs the query read-only. An error comes back as text, not as a crash.

    allowed_tables: None — any table; otherwise only these may be read, which
    the database itself enforces (table_guard).
    """
    if not is_safe_select(sql):
        return {"columns": [], "rows": [], "error": "Only SELECT queries are allowed"}

    try:
        connection = connect_existing(db_path, read_only=True)
    except (FileNotFoundError, sqlite3.Error) as exc:
        return {"columns": [], "rows": [], "error": str(exc)}
    try:
        connection.text_factory = str
        # SQLite по умолчанию считает "неизвестное_имя" не ошибкой, а строковой
        # константой. Для SQL, написанного моделью, это худший из исходов:
        # опечатка в имени колонки вернула бы колонку одинаковых строк вместо
        # ошибки. Отключаем поблажку — пусть падает.
        if hasattr(connection, "setconfig"):
            connection.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, False)
        if allowed_tables is not None:
            connection.set_authorizer(table_guard(allowed_tables))
        cursor = connection.execute(with_limit(sql))
        columns = [description[0] for description in cursor.description or []]
        rows = [list(row) for row in cursor.fetchall()]
        return {"columns": columns, "rows": rows, "error": None}
    except sqlite3.Error as exc:
        message = str(exc)
        if allowed_tables is not None and ("not authorized" in message or "prohibited" in message):
            # The SQLite message names the closed table and column; it must not
            # tell a restricted user what the tables they cannot read hold.
            message = "not authorized: a table this account may not read"
        return {"columns": [], "rows": [], "error": message}
    finally:
        connection.close()


def rows_for_prompt(result: dict[str, Any]) -> str:
    header = " | ".join(result["columns"])
    body = "\n".join(
        " | ".join("" if v is None else str(v) for v in row)
        for row in result["rows"][:ROWS_IN_PROMPT]
    )
    more = ""
    if len(result["rows"]) > ROWS_IN_PROMPT:
        more = f"\n… {len(result['rows']) - ROWS_IN_PROMPT} more rows"
    return f"{header}\n{body}{more}" if body else "(empty result)"

