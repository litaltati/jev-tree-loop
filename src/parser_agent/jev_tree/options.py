"""What the decision tree can choose from — read from the passports, never typed in.

A table fits the tree when it has a row-label column (the lines of the report)
and numeric columns whose names break into parts:

    acmetel_may_actual_2026            entity "acmetel" · period "may" · measure "actual" · 2026
    group_ytd_budget_2026              entity "group"   · period "ytd" · measure "budget" · 2026
    standalone_format_x_may_actual_2026  entity "standalone format x" · …

Only the generic words of a report are known in advance: the months, "ytd",
actual / budget and their short forms. The entities, the lines and the tables
come from the data, so a new workbook brings its own options with it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..catalog import connect_existing, quote_identifier
from ..chat import hierarchy_levels

MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
PERIODS = MONTHS + ("ytd",)
MEASURES = {"actual": "actual", "act": "actual", "fact": "actual",
            "budget": "budget", "bud": "budget", "bdgt": "budget", "plan": "budget"}
_YEAR = re.compile(r"^(19|20)\d\d$")
# Parts that compare two figures ("act_vs_bud") rather than hold one.
_VARIANCE = {"vs", "diff", "variance", "delta"}
_NUMERIC_TYPES = {"REAL", "INTEGER", "NUMERIC", "FLOAT"}


@dataclass(frozen=True)
class ColumnParts:
    column: str
    entity: str          # "" when the table has a single entity
    period: str          # a month or "ytd"
    measure: str         # "actual" or "budget"
    year: int | None
    duplicate: bool      # "_2", "_3": a second copy of a header


def parse_column(name: str) -> ColumnParts | None:
    """The parts of a numeric column name; None when it is not entity × period × measure."""
    tokens = name.lower().split("_")
    if any(t in _VARIANCE for t in tokens):
        return None
    period_at = next((i for i, t in enumerate(tokens) if t in PERIODS), None)
    measure_at = next((i for i, t in enumerate(tokens) if t in MEASURES), None)
    if period_at is None or measure_at is None:
        return None
    year = next((int(t) for t in tokens if _YEAR.match(t)), None)
    duplicate = tokens[-1].isdigit() and not _YEAR.match(tokens[-1]) and len(tokens[-1]) <= 2
    entity_tokens = [t for t in tokens[: min(period_at, measure_at)] if t != "col" and not _YEAR.match(t)]
    return ColumnParts(
        column=name, entity=" ".join(entity_tokens), period=tokens[period_at],
        measure=MEASURES[tokens[measure_at]], year=year, duplicate=duplicate,
    )


@dataclass
class TableOptions:
    name: str
    description: str
    label_column: str
    level_column: str | None                 # the group a row belongs to, when rows are nested
    labels: list[str]
    columns: list[ColumnParts] = field(default_factory=list)
    # From the rows themselves: which group a line belongs to and which lines
    # a group is made of (when the lines are nested), and the names that occur
    # on more than one row of the report.
    groups: dict[str, list[str]] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=dict)
    duplicates: set[str] = field(default_factory=set)

    def describe(self, label: str) -> str | None:
        """What a line is, in words Jev can use: "Salaries — part of Personnel costs"."""
        parts = []
        if self.groups.get(label):
            parts.append("part of " + " and of ".join(self.groups[label]))
        if self.children.get(label):
            kids = self.children[label]
            parts.append("made of " + ", ".join(kids[:8]) + (" …" if len(kids) > 8 else ""))
        return "; ".join(parts) or None

    @property
    def entities(self) -> list[str]:
        return sorted({c.entity for c in self.columns})

    @property
    def periods(self) -> list[str]:
        return [p for p in PERIODS if any(c.period == p for c in self.columns)]

    def pick_columns(self, entities: list[str], period: str | None, measure: str | None) -> list[ColumnParts]:
        """The columns for these entities. A period or measure that was not said
        means: every period the table has, actual figures only."""
        chosen = []
        for entity in entities:
            own = [c for c in self.columns if c.entity == entity and not c.duplicate]
            if not own:
                continue
            latest = max((c.year or 0) for c in own)
            for c in own:
                if (c.year or 0) != latest:
                    continue
                if period and c.period != period:
                    continue
                if c.measure != (measure or "actual"):
                    continue
                chosen.append(c)
        return chosen


def table_options(catalog: dict[str, Any]) -> TableOptions | None:
    """The options of one table, or None when the tree cannot answer from it."""
    columns = catalog.get("columns") or []
    names = [str(c.get("column_name", "")) for c in columns]
    levels = hierarchy_levels(names)
    text_columns = [c for c in columns if c.get("allowed_data") and str(c.get("column_data_type", "")).upper() == "TEXT"]
    if not text_columns:
        return None
    label_names = {label for label, _ in levels.values()}
    label = next((c for c in text_columns if c["column_name"] in label_names), None)
    if label is None:
        label = max(text_columns, key=lambda c: len(c.get("allowed_data") or []))
    level = next((lc for lc, (lab, n) in levels.items() if lab == label["column_name"] and n == 0), None)
    parsed = [
        p for c in columns
        if str(c.get("column_data_type", "")).upper() in _NUMERIC_TYPES
        and (p := parse_column(str(c.get("column_name", "")))) is not None
    ]
    if not parsed:
        return None
    labels = [str(v) for v in label["allowed_data"] if v is not None and str(v).strip() and str(v) != "Null"]
    return TableOptions(
        name=catalog["table_name"], description=str(catalog.get("table_description", "")),
        label_column=label["column_name"], level_column=level, labels=labels, columns=parsed,
    )


def read_hierarchy(options: TableOptions, db_path: Path) -> None:
    """Fills groups, children and duplicates from the table's rows."""
    label = quote_identifier(options.label_column)
    connection = connect_existing(db_path, read_only=True)
    try:
        counts = connection.execute(
            f"SELECT {label}, COUNT(*) FROM {quote_identifier(options.name)} GROUP BY {label} HAVING COUNT(*) > 1"
        ).fetchall()
        pairs = [] if options.level_column is None else connection.execute(
            f"SELECT DISTINCT {label}, {quote_identifier(options.level_column)} FROM {quote_identifier(options.name)}"
        ).fetchall()
    finally:
        connection.close()
    options.duplicates = {str(name) for name, _ in counts if name is not None}
    for line, group in pairs:
        if line is None or group is None:
            continue
        line, group = str(line), str(group)
        if line != group:
            options.groups.setdefault(line, []).append(group)
            options.children.setdefault(group, []).append(line)


def tree_options(catalogs: dict[str, dict[str, Any]], db_path: Path | None = None) -> dict[str, TableOptions]:
    """Every table the tree can answer from, by name. With the database, the
    nesting of the lines is read as well."""
    out = {}
    for name, catalog in catalogs.items():
        options = table_options(catalog)
        if options is None:
            continue
        if db_path is not None:
            read_hierarchy(options, db_path)
        out[name] = options
    return out
