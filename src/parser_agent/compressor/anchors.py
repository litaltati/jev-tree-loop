"""Module 1: Structural-Anchor-Based Extraction (SpreadsheetLLM, arXiv:2407.09025).

The idea of the paper: in real sheets most rows/columns are homogeneous (solid
data), while the informative ones are the "structural anchors" — rows/columns
where the content type or the formatting changes: headers, block boundaries,
total rows. We keep the anchors + k neighbours on each side, drop the rest and
renumber the coordinates, keeping a mapping back to the original Excel ones.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Iterator

from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

# Protection against sheets with a "junk" max_row/max_column (Excel often keeps
# formatting down to row 1048576).
MAX_ROWS = 20_000
MAX_COLS = 256

# Heterogeneity threshold: the share of positions in a row/column that must
# differ from the neighbour for it to count as an anchor. 0.25 was picked
# empirically — headers and block boundaries differ from their neighbours by at
# least a quarter of the cells.
DEFAULT_THRESHOLD = 0.25

# How many rows/columns to keep around each anchor (k from the paper).
DEFAULT_K = 4


@dataclass(frozen=True)
class CellInfo:
    """Everything worth knowing about a cell for compression (value + format + styling)."""

    row: int                      # original Excel row (1-based)
    col: int                      # original Excel column (1-based)
    value: object = None
    number_format: str = "General"
    is_formula: bool = False
    is_merged: bool = False       # belongs to a merged range
    is_merge_origin: bool = False  # top-left cell of a merged range
    bold: bool = False
    has_border: bool = False
    has_fill: bool = False

    @property
    def is_empty(self) -> bool:
        return self.value is None or (isinstance(self.value, str) and not self.value.strip())

    @property
    def address(self) -> str:
        """Address in the ORIGINAL Excel coordinates."""
        return f"{get_column_letter(self.col)}{self.row}"


@dataclass
class CompactSheet:
    """The compressed sheet matrix + a mapping from new coordinates to original ones."""

    sheet_name: str
    cells: dict[tuple[int, int], CellInfo] = field(default_factory=dict)  # (new_row, new_col) -> cell
    row_map: dict[int, int] = field(default_factory=dict)   # new_row -> original_row
    col_map: dict[int, int] = field(default_factory=dict)   # new_col -> original_col
    n_rows: int = 0
    n_cols: int = 0
    original_n_rows: int = 0
    original_n_cols: int = 0
    merged_ranges: list[str] = field(default_factory=list)  # in the original coordinates

    def original_coords(self, new_row: int, new_col: int) -> tuple[int, int]:
        return self.row_map[new_row], self.col_map[new_col]

    def original_address(self, new_row: int, new_col: int) -> str:
        row, col = self.original_coords(new_row, new_col)
        return f"{get_column_letter(col)}{row}"

    def iter_cells(self) -> Iterator[tuple[int, int, CellInfo]]:
        """Non-empty cells in reading order (top to bottom, left to right)."""
        for (new_row, new_col) in sorted(self.cells):
            yield new_row, new_col, self.cells[(new_row, new_col)]

    def by_original_address(self) -> dict[str, CellInfo]:
        """An index `"B7" -> CellInfo` for the aggregation module."""
        return {info.address: info for info in self.cells.values()}

    @property
    def kept_rows(self) -> list[int]:
        return [self.row_map[i] for i in sorted(self.row_map)]

    @property
    def kept_cols(self) -> list[int]:
        return [self.col_map[i] for i in sorted(self.col_map)]


def _type_code(value: object) -> int:
    """A coarse type classification used to compare adjacent rows/columns."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return 0
    if isinstance(value, bool):
        return 4
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return 3
    if isinstance(value, (int, float)):
        return 2
    return 1


def _read_cell(cell, merged_all: set[tuple[int, int]],
               merged_origins: set[tuple[int, int]]) -> CellInfo:
    coords = (cell.row, cell.column)
    value = cell.value

    # The workbook is read with data_only=True, so a formula arrives as its
    # computed value and the flag stays False. That is the point: what the sheet
    # means is in the values, not in the formulas behind them.
    is_formula = cell.data_type == "f"

    font = cell.font
    border = cell.border
    fill = cell.fill
    has_border = any(
        getattr(border, side).style is not None
        for side in ("left", "right", "top", "bottom")
    )
    has_fill = bool(
        fill is not None and fill.fill_type not in (None, "none")
    )

    return CellInfo(
        row=cell.row,
        col=cell.column,
        value=value,
        number_format=cell.number_format or "General",
        is_formula=is_formula,
        is_merged=coords in merged_all,
        is_merge_origin=coords in merged_origins,
        bold=bool(font is not None and font.bold),
        has_border=has_border,
        has_fill=has_fill,
    )


def read_grid(ws: Worksheet) -> list[list[CellInfo]]:
    """Reads the sheet into a dense CellInfo matrix (1-based coordinates kept inside)."""
    n_rows = min(ws.max_row or 1, MAX_ROWS)
    n_cols = min(ws.max_column or 1, MAX_COLS)

    merged_all: set[tuple[int, int]] = set()
    merged_origins: set[tuple[int, int]] = set()
    for rng in ws.merged_cells.ranges:
        merged_origins.add((rng.min_row, rng.min_col))
        if (rng.max_row - rng.min_row + 1) * (rng.max_col - rng.min_col + 1) > 100_000:
            continue  # a merge spanning the whole sheet — do not expand it
        for row in range(rng.min_row, rng.max_row + 1):
            for col in range(rng.min_col, rng.max_col + 1):
                merged_all.add((row, col))

    grid: list[list[CellInfo]] = []
    for row_cells in ws.iter_rows(min_row=1, max_row=n_rows, min_col=1, max_col=n_cols):
        row_info: list[CellInfo] = []
        for cell in row_cells:
            row_info.append(_read_cell(cell, merged_all, merged_origins))
        grid.append(row_info)
    return grid


def _descriptor(info: CellInfo) -> tuple:
    """The cell features by which heterogeneity is measured."""
    return (
        _type_code(info.value),
        info.bold,
        info.has_border,
        info.has_fill,
        info.is_merged,
        info.is_formula,
    )


def _diff(a: list[tuple], b: list[tuple]) -> float:
    """The share of positions in which two feature vectors differ."""
    if not a and not b:
        return 0.0
    size = max(len(a), len(b))
    empty = (0, False, False, False, False, False)
    mismatches = sum(
        1
        for i in range(size)
        if (a[i] if i < len(a) else empty) != (b[i] if i < len(b) else empty)
    )
    return mismatches / size


def _find_anchors(
    vectors: list[list[tuple]],
    threshold: float,
    forced: set[int] | None = None,
) -> set[int]:
    """Indexes (1-based) of rows/columns with high heterogeneity.

    `forced` — indexes that become anchors unconditionally (rows and columns
    that start a merged range: hierarchical labels sit there, and such a row
    differs from its neighbours by only one cell, so it fails the threshold).
    """
    count = len(vectors)
    if count == 0:
        return set()

    empty = [(0, False, False, False, False, False)] * (len(vectors[0]) if vectors else 0)
    anchors: set[int] = set()
    non_empty: list[int] = []

    for i in range(count):
        current = vectors[i]
        if any(desc[0] != 0 for desc in current):
            non_empty.append(i + 1)
        prev_vec = vectors[i - 1] if i > 0 else empty
        next_vec = vectors[i + 1] if i + 1 < count else empty
        # max, not the average: a block boundary differs from at least one neighbour.
        score = max(_diff(current, prev_vec), _diff(current, next_vec))
        if score >= threshold:
            anchors.add(i + 1)

    # The first and last non-empty rows/columns are always the table boundaries.
    if non_empty:
        anchors.add(non_empty[0])
        anchors.add(non_empty[-1])
    if forced:
        anchors.update(i for i in forced if 1 <= i <= count)
    return anchors


def _keep_around(anchors: set[int], k: int, limit: int) -> list[int]:
    """Anchors + k neighbours on each side, sorted and deduplicated."""
    kept: set[int] = set()
    for anchor in anchors:
        for i in range(anchor - k, anchor + k + 1):
            if 1 <= i <= limit:
                kept.add(i)
    return sorted(kept)


def build_compact_sheet(
    ws: Worksheet,
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> CompactSheet:
    """Builds a CompactSheet: anchors + k neighbours, with renumbered coordinates."""
    grid = read_grid(ws)
    n_rows = len(grid)
    n_cols = len(grid[0]) if grid else 0

    if n_rows == 0 or n_cols == 0:
        return CompactSheet(sheet_name=ws.title)

    row_vectors = [[_descriptor(info) for info in row] for row in grid]
    col_vectors = [[_descriptor(grid[r][c]) for r in range(n_rows)] for c in range(n_cols)]

    # The starts of merged ranges are anchors without a threshold check.
    merge_origin_rows = {info.row for row in grid for info in row if info.is_merge_origin}
    merge_origin_cols = {info.col for row in grid for info in row if info.is_merge_origin}

    kept_rows = _keep_around(
        _find_anchors(row_vectors, threshold, merge_origin_rows), k, n_rows
    )
    kept_cols = _keep_around(
        _find_anchors(col_vectors, threshold, merge_origin_cols), k, n_cols
    )

    # Degenerate case (a sheet without anchors) — keep everything so no data is lost.
    if not kept_rows:
        kept_rows = list(range(1, n_rows + 1))
    if not kept_cols:
        kept_cols = list(range(1, n_cols + 1))

    compact = CompactSheet(
        sheet_name=ws.title,
        row_map={i: orig for i, orig in enumerate(kept_rows, start=1)},
        col_map={i: orig for i, orig in enumerate(kept_cols, start=1)},
        n_rows=len(kept_rows),
        n_cols=len(kept_cols),
        original_n_rows=n_rows,
        original_n_cols=n_cols,
        merged_ranges=[str(rng) for rng in ws.merged_cells.ranges],
    )

    for new_row, orig_row in compact.row_map.items():
        for new_col, orig_col in compact.col_map.items():
            info = grid[orig_row - 1][orig_col - 1]
            if not info.is_empty:  # empty cells are not stored in the compact matrix
                compact.cells[(new_row, new_col)] = info

    return compact
