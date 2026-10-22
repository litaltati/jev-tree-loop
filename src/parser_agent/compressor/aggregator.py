"""Module 3: Data-Format-Aware Aggregation (SpreadsheetLLM, arXiv:2407.09025).

The concrete numbers/dates are not needed to recognize the structure of a sheet
— it is enough to know that D2:D145 holds a currency format. So all "format"
cells are replaced with a single entry per format, while text (headers and row
labels, from which the LLM takes the column names) is left as is.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections import defaultdict

from openpyxl.utils import column_index_from_string, get_column_letter

from .anchors import CellInfo, CompactSheet
from .inverted import fold_addresses

# Categories that are NOT aggregated: the LLM takes the column names from them.
NON_AGGREGATED = {"Text", "Empty"}

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_SCIENTIFIC_RE = re.compile(r"^[+-]?\d+(\.\d+)?[eE][+-]?\d+$")
_PERCENT_RE = re.compile(r"^[+-]?\d+([.,]\d+)?\s*%$")
_CURRENCY_RE = re.compile(r"^[$€₽£¥]\s?[+-]?[\d\s.,]+$|^[+-]?[\d\s.,]+\s?(руб\.?|₽|\$|€|USD|EUR|RUB)$", re.IGNORECASE)
_INT_RE = re.compile(r"^[+-]?\d{1,3}(\s\d{3})*$|^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?\d+[.,]\d+$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$|^\d{1,2}[./]\d{1,2}[./]\d{2,4}$")
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")

_CURRENCY_TOKENS = ("$", "€", "₽", "£", "¥", "руб", "USD", "EUR", "RUB", "[$")


def nfs_category(number_format: str) -> str | None:
    """The category by Number Format String; None if the format says nothing."""
    fmt = (number_format or "").strip()
    if not fmt or fmt.lower() == "general":
        return None
    if "%" in fmt:
        return "Percentage"
    if any(token in fmt for token in _CURRENCY_TOKENS):
        return "Currency"
    if re.search(r"[EeЕ]\+", fmt):
        return "Scientific"
    lowered = fmt.lower()
    has_date = any(token in lowered for token in ("yy", "mmm", "dd", "d/", "m/", "yyyy"))
    has_time = any(token in lowered for token in ("h:", "hh", "ss", "am/pm"))
    if has_date:
        return "Date"
    if has_time:
        return "Time"
    if fmt == "@":
        return "Text"
    if re.search(r"[0#]", fmt):
        return "Float" if ("." in fmt or "0.0" in fmt) else "Integer"
    return None


def detect_type(value: object, number_format: str = "General") -> str:
    """A rule-based value type for when the NFS is missing/useless.

    Returns one of: Integer, Float, Date, Percentage, Currency, Email,
    Year, Time, Scientific, Text, Empty.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return "Empty"
    if isinstance(value, bool):
        return "Text"  # booleans are treated as labels, not as numbers
    if isinstance(value, dt.datetime):
        return "Date"
    if isinstance(value, dt.date):
        return "Date"
    if isinstance(value, dt.time):
        return "Time"
    if isinstance(value, int):
        # A 4-digit integer in a sensible range is almost always a year (column "2023").
        if 1900 <= value <= 2100:
            return "Year"
        return "Integer"
    if isinstance(value, float):
        return "Integer" if value.is_integer() else "Float"

    text = str(value).strip()
    if _EMAIL_RE.match(text):
        return "Email"
    if _PERCENT_RE.match(text):
        return "Percentage"
    if _CURRENCY_RE.match(text):
        return "Currency"
    if _SCIENTIFIC_RE.match(text):
        return "Scientific"
    if _TIME_RE.match(text):
        return "Time"
    if _DATE_RE.match(text):
        return "Date"
    if _INT_RE.match(text):
        return "Year" if re.fullmatch(r"(19|20)\d{2}", text) else "Integer"
    if _FLOAT_RE.match(text):
        return "Float"
    return "Text"


def cell_category(info: CellInfo) -> tuple[str, str]:
    """(aggregation key, category). The key is the NFS itself if it is informative."""
    category = nfs_category(info.number_format)
    if category is not None and category != "Text":
        return info.number_format.strip(), category
    detected = detect_type(info.value, info.number_format)
    return detected, detected


def _parse_addresses(spec: str) -> list[tuple[int, int]]:
    """Expands "A1:A3,C5" back into a list of (row, col)."""
    coords: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        start, _, end = part.partition(":")
        match_start = re.match(r"([A-Z]{1,3})(\d+)$", start)
        if not match_start:
            continue
        c1 = column_index_from_string(match_start.group(1))
        r1 = int(match_start.group(2))
        if not end:
            coords.append((r1, c1))
            continue
        match_end = re.match(r"([A-Z]{1,3})(\d+)$", end)
        if not match_end:
            coords.append((r1, c1))
            continue
        c2 = column_index_from_string(match_end.group(1))
        r2 = int(match_end.group(2))
        for row in range(min(r1, r2), max(r1, r2) + 1):
            for col in range(min(c1, c2), max(c1, c2) + 1):
                coords.append((row, col))
    return coords


def aggregate(inverted_json: str, compact: CompactSheet) -> str:
    """Folds the format cells of the inverted index into entries such as
    `{"$#,##0.00": "D2:D145"}`, leaving text entries unchanged."""
    index: dict[str, str] = json.loads(inverted_json)
    cells = compact.by_original_address()

    text_entries: dict[str, str] = {}
    coords_by_format: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for value, spec in index.items():
        kept_coords: list[tuple[int, int]] = []
        for row, col in _parse_addresses(spec):
            info = cells.get(f"{get_column_letter(col)}{row}")
            if info is None:
                continue
            key, category = cell_category(info)
            if category in NON_AGGREGATED:
                kept_coords.append((row, col))  # text is not aggregated
            else:
                coords_by_format[key].append((row, col))
        if kept_coords:
            text_entries[value] = fold_addresses(kept_coords)

    # Order: text first (headers/labels), then the format blocks.
    result: dict[str, str] = dict(text_entries)
    for key, coords in coords_by_format.items():
        folded = fold_addresses(coords)
        if key in result:
            # A rare collision between a text value and a format name — the
            # semantics are the same ("these addresses"), so we just merge them.
            result[key] = f"{result[key]},{folded}"
        else:
            result[key] = folded

    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))
