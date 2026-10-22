"""Stage 4: the passport of a finished table — a data catalog entry in JSON.

The file is what a text-to-SQL model reads instead of the database itself: the
column names, their types, an example value, the values that actually occur, and
prose describing what all of it means.

Three kinds of fields, three different sources:

  * `table_name`, `column_name`, `column_data_type` — read from the database
    itself (`sqlite_master`, `PRAGMA table_info`);
  * `sample_data`, `allowed_data` — counted by SQL over the loaded rows;
  * `table_description`, `column_description` — written by the LLM.

The model is never asked for a value a query could be built on: everything of
that kind is computed from the data, and the model only writes the prose. A
value it invented would send the query looking for rows that do not exist.

The catalog is always generated from scratch — it is not a file to be edited by
hand.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .routing import DEFAULT_ROUTES, Task, resolve_model

logger = logging.getLogger(__name__)

DEFAULT_MODEL = DEFAULT_ROUTES[Task.CATALOG]

# One schema for every table the tool produces: SQLite has no schemas of its
# own, and the field exists for the catalog the tables are published into.
DEFAULT_SCHEMA_NAME = "gold_data_sets"

# How an absent value is written inside `allowed_data`. A label, not a value:
# NULL cannot be written into a list of strings as itself.
NULL_LABEL = "Null"

# How many values of one column are shown to the model. The file itself keeps
# all of them — this limit only keeps the prompt from growing to the size of
# the table.
MAX_VALUES_IN_PROMPT = 30


@dataclass
class ColumnFacts:
    """What the database itself knows about one column."""

    column_name: str
    column_data_type: str            # the type as declared in the table
    sample_data: str = ""            # one real value, as text
    allowed_data: list[str] = field(default_factory=list)  # text columns only

    def to_json(self, description: str = "") -> dict[str, Any]:
        return {
            "column_name": self.column_name,
            "column_data_type": self.column_data_type,
            "sample_data": self.sample_data,
            "allowed_data": list(self.allowed_data),
            "column_description": description,
        }


@dataclass
class TableFacts:
    """The deterministic part of the passport: everything except the prose."""

    table_name: str
    schema_name: str
    columns: list[ColumnFacts] = field(default_factory=list)
    row_count: int = 0

    def to_json(
        self,
        table_description: str = "",
        column_descriptions: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """The finished passport. The descriptions are filled in by the model."""
        descriptions = column_descriptions or {}
        return {
            "table_name": self.table_name,
            "schema_name": self.schema_name,
            "table_description": table_description,
            "columns": [
                column.to_json(descriptions.get(column.column_name, ""))
                for column in self.columns
            ],
        }


def connect_existing(db_path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Opens a database that must already exist.

    SQLite has no separate «create» step: connecting to a path that is not there
    silently makes an empty database at that path. A typo in the path then does
    not fail — it succeeds, and the mistake surfaces one step later as «no such
    table», pointing at the missing table instead of at the wrong file.

    So the file is checked first. Where a database is meant to be created — the
    extractor writing its output — this is not used.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database {path} not found — check the path")
    if read_only:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    return sqlite3.connect(path)


def quote_identifier(name: str) -> str:
    """A table or column name for SQL: double quotes, the inner ones doubled."""
    return '"' + name.replace('"', '""') + '"'


def is_text_type(declared_type: str) -> bool:
    """Whether the column has TEXT affinity, by the SQLite rule.

    The rule is the one SQLite itself applies: a declared type containing CHAR,
    CLOB or TEXT gets TEXT affinity. `allowed_data` is listed for such columns
    only — for the rest the field stays empty.
    """
    text = (declared_type or "").upper()
    return any(mark in text for mark in ("CHAR", "CLOB", "TEXT"))


def list_tables(connection: sqlite3.Connection) -> list[str]:
    """The tables of the database, without SQLite's own service tables."""
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [row[0] for row in rows]


def _as_text(value: object) -> str:
    """A cell value as text for the passport."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return str(value)


def sample_value(connection: sqlite3.Connection, table: str, column: str) -> str:
    """The first value of the column that is not NULL. Empty if there are none."""
    row = connection.execute(
        f"SELECT {quote_identifier(column)} FROM {quote_identifier(table)} "
        f"WHERE {quote_identifier(column)} IS NOT NULL LIMIT 1"
    ).fetchone()
    return _as_text(row[0]) if row is not None else ""


def distinct_values(connection: sqlite3.Connection, table: str, column: str) -> list[str]:
    """Every value of the column, each one once, the frequent ones first.

    There is no cap: for a text column the whole domain goes into the passport,
    however large it is. NULL and an empty cell are one and the same thing for
    the author of a query, and both are written as a single NULL_LABEL.
    """
    rows = connection.execute(
        f"SELECT {quote_identifier(column)}, COUNT(*) AS n "
        f"FROM {quote_identifier(table)} GROUP BY 1 ORDER BY n DESC, 1"
    ).fetchall()

    values: list[str] = []
    has_null = False
    for value, _count in rows:
        text = _as_text(value)
        if not text:
            has_null = True
            continue
        if text not in values:
            values.append(text)
    return ([NULL_LABEL] if has_null else []) + values


def table_facts(
    connection: sqlite3.Connection,
    table: str,
    *,
    schema_name: str = DEFAULT_SCHEMA_NAME,
) -> TableFacts:
    """Reads everything about the table that does not need a model."""
    info = connection.execute(
        f"PRAGMA table_info({quote_identifier(table)})"
    ).fetchall()
    if not info:
        raise KeyError(f"Таблица {table!r} не найдена в базе")

    row_count = connection.execute(
        f"SELECT COUNT(*) FROM {quote_identifier(table)}"
    ).fetchone()[0]

    columns: list[ColumnFacts] = []
    for row in info:
        name, declared = row[1], (row[2] or "").strip().upper()
        columns.append(
            ColumnFacts(
                column_name=name,
                column_data_type=declared,
                sample_data=sample_value(connection, table, name),
                allowed_data=(
                    distinct_values(connection, table, name)
                    if is_text_type(declared)
                    else []
                ),
            )
        )
    return TableFacts(
        table_name=table, schema_name=schema_name, columns=columns, row_count=row_count
    )


# How many of a sibling's own columns are shown as its fingerprint.
SIBLING_COLUMNS = 6


def sibling_fingerprints(
    connection: sqlite3.Connection, table: str, limit: int = 8
) -> list[dict[str, Any]]:
    """What the neighbouring tables of the same database look like.

    One workbook makes sheets that are alike: the same report in another format,
    for another company, for another period. Described one by one, they all come
    out as «financial data» and become indistinguishable — in the catalog and in
    any search over it. So the model is shown its neighbours and what it does
    not share with them.
    """
    own = {row[1] for row in connection.execute(f"PRAGMA table_info({quote_identifier(table)})")}
    siblings: list[dict[str, Any]] = []
    for name in list_tables(connection):
        if name == table:
            continue
        columns = [
            row[1]
            for row in connection.execute(f"PRAGMA table_info({quote_identifier(name)})")
        ]
        rows = connection.execute(
            f"SELECT COUNT(*) FROM {quote_identifier(name)}"
        ).fetchone()[0]
        siblings.append(
            {
                "table_name": name,
                "rows": rows,
                "columns": len(columns),
                "own_columns": [c for c in columns if c not in own][:SIBLING_COLUMNS],
            }
        )
    return siblings[:limit]


SYSTEM_PROMPT = """You write data-catalog entries for tables in a SQLite database.
The entry is read by another model that writes SQL against this table and never sees
the data itself, so write what such a model needs in order not to guess.

You are given the facts of one table: its columns, the SQL type of each column, one
real sample value and, for text columns, the values that actually occur in the data.
A value listed as "Null" is a label for an empty cell, not a literal string.

Write in English:
- table_description: START by saying what makes THIS table different from its sibling
  tables — the entity, the report format, the level of detail, the period. Sibling
  sheets of one workbook look alike, and a description that would fit any of them is
  useless: never open with a generic sentence such as "This table holds financial
  data". Only after that say what the table holds, what a single row is (its
  granularity), which columns identify a row, which are measures and which are
  labels, and in what units.
- column_description for every column: what the value means and in what units. When
  the list of occurring values is short, explain what those values stand for.

Use ONLY the facts you are given. Do not invent business meaning, data sources,
owners, standard filters or metrics that the facts do not show. Do not repeat the
column name back as its description. If a column is unclear, describe it by its
type and its values.

Return ONLY valid JSON, no explanations, no markdown."""

_JSON_SHAPE = """{
  "table_description": "<what the table holds>",
  "columns": [
    {"column_name": "<exactly as given>", "column_description": "<what the column holds>"}
  ]
}"""


def _values_line(column: ColumnFacts) -> str:
    """The value list of a column for the prompt, shortened to a readable size."""
    if not column.allowed_data:
        return ""
    shown = column.allowed_data[:MAX_VALUES_IN_PROMPT]
    rest = len(column.allowed_data) - len(shown)
    listed = ", ".join(shown)
    return f" | values ({len(column.allowed_data)}): {listed}" + (
        f", ... +{rest} more" if rest else ""
    )


def build_user_prompt(
    facts: TableFacts,
    *,
    sheet_name: str | None = None,
    chains: dict[str, str] | None = None,
    notes: str = "",
    siblings: list[dict[str, Any]] | None = None,
) -> str:
    """The user message: the facts of the table and the shape of the answer.

    `chains` are the glued header cells of the source sheet, by column name. In
    the name the meaning is already squeezed to fit SQL; in the chain the words
    of the sheet are still there, so it is the chain that a description should
    be written from.
    """
    lines = [
        f"Table: {facts.table_name} (schema {facts.schema_name}), "
        f"{facts.row_count} rows, {len(facts.columns)} columns."
    ]
    if sheet_name:
        lines.append(f'Source: sheet "{sheet_name}" of an Excel workbook.')
    if notes:
        lines.append(f"Observations about the sheet structure: {notes}")

    lines.append("\nColumns:")
    for column in facts.columns:
        line = (
            f"- {column.column_name} | {column.column_data_type or 'unknown type'}"
            f" | sample: {column.sample_data or '(empty)'}"
        )
        lines.append(line + _values_line(column))

    if chains:
        lines.append(
            "\nHeader cells of the source sheet, glued top to bottom (`__` separates "
            "the header levels). The words of the sheet live here — write the "
            "descriptions from them:"
        )
        for column in facts.columns:
            chain = chains.get(column.column_name)
            if chain:
                lines.append(f"- {column.column_name}: {chain}")

    if siblings:
        lines.append(
            "\nOther tables in the same database. They came from sheets of the same "
            "workbook and look alike, so the description of THIS table must make it "
            "distinguishable from every one of them:"
        )
        for sibling in siblings:
            own = ", ".join(sibling["own_columns"]) or "no columns of its own"
            lines.append(
                f"- {sibling['table_name']}: {sibling['rows']} rows, "
                f"{sibling['columns']} columns; columns this table does NOT have: {own}"
            )

    lines.append(
        "\nReturn JSON of exactly this shape, one entry per column, in the same "
        "order:\n" + _JSON_SHAPE
    )
    return "\n".join(lines)


def parse_descriptions(raw_text: str) -> tuple[str, dict[str, str]]:
    """Parses the model answer into (table description, {column: description})."""
    from .describer import _strip_code_fence

    payload = json.loads(_strip_code_fence(raw_text))
    if not isinstance(payload, dict):
        raise ValueError("Ожидался JSON-объект с описаниями")

    table_description = str(payload.get("table_description") or "").strip()
    columns: dict[str, str] = {}
    for item in payload.get("columns") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("column_name") or "").strip()
        if name:
            columns[name] = str(item.get("column_description") or "").strip()
    if not table_description and not columns:
        raise ValueError("В ответе нет ни описания таблицы, ни описаний колонок")
    return table_description, columns


def _default_client() -> Any:
    """Lazy openai import: the package is only needed for a real API call."""
    from openai import AsyncOpenAI

    return AsyncOpenAI()


async def describe_table(
    facts: TableFacts,
    *,
    sheet_name: str | None = None,
    chains: dict[str, str] | None = None,
    notes: str = "",
    siblings: list[dict[str, Any]] | None = None,
    model: str | None = None,
    client: Any | None = None,
    on_raw_response: Callable[[str], None] | None = None,
) -> tuple[str, dict[str, str]]:
    """Asks the model for the prose of the passport.

    On a failed parse one retry is made, exactly as at stage 2: the model is
    given its own answer and the error, and asked for corrected JSON.
    """
    model = model or resolve_model(Task.CATALOG)
    client = client or _default_client()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                facts, sheet_name=sheet_name, chains=chains, notes=notes,
                siblings=siblings,
            ),
        },
    ]

    last_error: Exception | None = None
    for attempt in range(2):
        response = await client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
        raw_text = response.choices[0].message.content or ""
        if on_raw_response is not None:
            on_raw_response(raw_text)
        try:
            return parse_descriptions(raw_text)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "[%s] описания не разобраны (попытка %d): %s",
                facts.table_name, attempt + 1, exc,
            )
            if attempt == 0:
                messages.append({"role": "assistant", "content": raw_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Твой ответ не удалось разобрать. Ошибка:\n"
                            f"{exc}\n\n"
                            "Верни ИСПРАВЛЕННЫЙ валидный JSON той же формы. "
                            "Только JSON, без markdown и пояснений."
                        ),
                    }
                )

    raise ValueError(
        f"Не удалось получить описания для таблицы {facts.table_name!r}: {last_error}"
    ) from last_error


async def build_catalog(
    db_path: str,
    table: str,
    *,
    schema_name: str = DEFAULT_SCHEMA_NAME,
    sheet_name: str | None = None,
    chains: dict[str, str] | None = None,
    notes: str = "",
    model: str | None = None,
    client: Any | None = None,
    describe: bool = True,
    on_raw_response: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """The finished passport of one table: the facts plus the prose.

    describe=False leaves the descriptions empty — the facts alone, without an
    API call.
    """
    connection = connect_existing(db_path)
    try:
        connection.text_factory = str
        facts = table_facts(connection, table, schema_name=schema_name)
        # The neighbours are read from the same database: it is what they have
        # in common that has to be told apart.
        siblings = sibling_fingerprints(connection, table)
    finally:
        connection.close()

    if not describe:
        return facts.to_json()

    table_description, column_descriptions = await describe_table(
        facts,
        sheet_name=sheet_name,
        chains=chains,
        notes=notes,
        siblings=siblings,
        model=model,
        client=client,
        on_raw_response=on_raw_response,
    )
    unknown = set(column_descriptions) - {c.column_name for c in facts.columns}
    if unknown:
        logger.warning(
            "[%s] модель описала колонки, которых нет в таблице: %s",
            table, ", ".join(sorted(unknown)),
        )
    return facts.to_json(table_description, column_descriptions)


def catalog_json(catalog: dict[str, Any]) -> str:
    """The passport as a file: UTF-8 as is, indented."""
    return json.dumps(catalog, ensure_ascii=False, indent=4) + "\n"


def build_catalog_sync(db_path: str, table: str, **kwargs: Any) -> dict[str, Any]:
    """A synchronous wrapper around build_catalog() for scripts and the REPL."""
    return asyncio.run(build_catalog(db_path, table, **kwargs))
