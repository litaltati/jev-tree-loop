"""Which tables actually hold what the question names.

The vector search answers «what is this table about». It cannot answer «where
does EBITDA live», and on sheets of one workbook that is the question that
decides everything: the same item and the same column exist in four tables at
once, and the passports describe all four in nearly the same words.

But the passports already carry the answer literally. `allowed_data` lists every
value of every text column — the row labels among them — and the column names are
right there too. So the question is matched against those lists, and the table is
chosen by what was found, not by what the description resembles.

No model and no vectors: this is a lookup in files that are already on disk.
"""

from __future__ import annotations

import re
from typing import Any

# Values shorter than this are not matched: "S", "FO" and the like would hit
# every question by accident.
MIN_VALUE_LEN = 3

# A found value is far stronger evidence than a shared word in a column name,
# and a table with nine "budget" columns must not win on that alone.
VALUE_WEIGHT = 2
MAX_COLUMN_POINTS = 3

_WORD = re.compile(r"[\w%]+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def _phrase(text: str) -> str:
    """The text as a space-padded chain of words — for whole-word matching."""
    words = _words(text)
    return f" {' '.join(words)} " if words else ""


def value_matches(question: str, catalog: dict[str, Any]) -> list[dict[str, str]]:
    """Values from the passport that the question names literally.

    Matching is by whole words: «ebitda» finds the item EBITDA, but the question
    «ebitda margin» finds both EBITDA and EBITDA MARGIN %, and the caller decides
    which is more precise.
    """
    haystack = _phrase(question)
    if not haystack:
        return []

    hits: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for column in catalog.get("columns") or []:
        name = column.get("column_name", "")
        for value in column.get("allowed_data") or []:
            text = str(value).strip()
            if len(text) < MIN_VALUE_LEN:
                continue
            needle = _phrase(text)
            if needle and needle in haystack and (name, text) not in seen:
                seen.add((name, text))
                hits.append({"column": name, "value": text})
    return hits


def column_matches(question: str, catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Columns whose name shares a word with the question.

    Column names are mechanical chains glued from a spreadsheet header, so a
    single word out of them — «budget», «may», «2026» — is a good pointer.
    """
    asked = set(_words(question))
    found: list[dict[str, Any]] = []
    for column in catalog.get("columns") or []:
        name = column.get("column_name", "")
        parts = {p for p in re.split(r"[_\W]+", name.lower()) if len(p) > 2}
        common = parts & asked
        if common:
            found.append({"column": name, "words": sorted(common)})
    return found


def _summary(values: list[dict[str, str]], columns: list[dict[str, Any]]) -> str:
    """One line for the choice card: what exactly was found in this table."""
    parts = []
    if values:
        shown = ", ".join(f'{v["value"]} → {v["column"]}' for v in values[:2])
        rest = f" +{len(values) - 2}" if len(values) > 2 else ""
        parts.append(shown + rest)
    if columns:
        names = ", ".join(c["column"] for c in columns[:2])
        rest = f" +{len(columns) - 2}" if len(columns) > 2 else ""
        parts.append(f"columns: {names}{rest}")
    return " · ".join(parts)


def collect_evidence(
    question: str, catalogs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Tables that hold what the question names, the best supported first."""
    found: list[dict[str, Any]] = []
    for name, catalog in catalogs.items():
        values = value_matches(question, catalog)
        columns = column_matches(question, catalog)
        if not values and not columns:
            continue
        score = VALUE_WEIGHT * len(values) + min(len(columns), MAX_COLUMN_POINTS)
        found.append(
            {
                "table_name": name,
                "table_description": catalog.get("table_description", ""),
                "values": values,
                "columns": columns,
                "score": score,
                "why": _summary(values, columns),
            }
        )
    return sorted(found, key=lambda e: (-e["score"], e["table_name"]))


def with_values(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only the tables where a named value was actually found.

    A shared word in a column name is a hint; a found value is a fact, and when
    there are facts the hints do not get a vote.
    """
    return [item for item in evidence if item["values"]]
