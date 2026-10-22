"""Module 0: bringing a spreadsheet to .xlsx before anything else reads it.

openpyxl, which every later stage reads the sheet with, understands .xlsx only.
A report saved as .xlsb, .xls, .ods and the like is first re-saved as .xlsx by
Aspose.Cells; an .xlsx goes on as it is, and Aspose is not even imported.

The converted file is temporary. openpyxl reads a workbook into memory in one
go and closes the file, so the copy is deleted as soon as the workbook is
loaded — compression, the model and the database all work from memory.

Aspose.Cells runs without a license, in evaluation mode. It then adds a
worksheet of its own ("Evaluation Warning") to every workbook it saves, and
opens at most 100 files per process. That worksheet is not part of the report:
only the sheets the source file had are handed on.
"""

from __future__ import annotations

import logging
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from openpyxl import Workbook, load_workbook

logger = logging.getLogger(__name__)

# The format every later stage reads.
NATIVE_SUFFIX = ".xlsx"

# What is converted through Aspose.Cells. CSV is deliberately absent: without
# an explicit separator Aspose guesses, and on a test file it split "A;1,5" at
# the comma and read the number "1.5" as the 1st of May.
CONVERTIBLE_SUFFIXES = frozenset({".xlsb", ".xls", ".xlsm", ".xltx", ".xltm", ".ods"})


class ConversionError(Exception):
    """The file cannot be brought to .xlsx: an unknown format or a failed conversion."""


@dataclass(frozen=True)
class PreparedWorkbook:
    """The .xlsx the pipeline reads, and which of its sheets came from the source."""

    path: Path                                    # the source itself or its temporary copy
    source_sheets: tuple[str, ...] | None = None  # None — every sheet of `path`

    @property
    def converted(self) -> bool:
        return self.source_sheets is not None

    def load(self) -> Workbook:
        """Opens the workbook with data_only=True: formulas arrive as their values."""
        if not self.converted:
            return load_workbook(self.path, data_only=True)
        # Aspose writes Excel extensions (conditional formatting and the like)
        # that openpyxl does not read and drops with a warning per sheet. They
        # carry nothing the pipeline uses.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*extension is not supported", category=UserWarning
            )
            return load_workbook(self.path, data_only=True)

    def sheet_names(self, workbook: Workbook) -> list[str]:
        """The sheets to process, in workbook order, without the evaluation sheet."""
        if self.source_sheets is None:
            return list(workbook.sheetnames)
        keep = set(self.source_sheets)
        return [name for name in workbook.sheetnames if name in keep]


def check_input(path: Path) -> None:
    """Refuses a format the pipeline cannot read, before any work is done."""
    suffix = path.suffix.lower()
    if suffix != NATIVE_SUFFIX and suffix not in CONVERTIBLE_SUFFIXES:
        known = ", ".join(sorted({NATIVE_SUFFIX, *CONVERTIBLE_SUFFIXES}))
        raise ConversionError(
            f"Формат {path.suffix or '(без расширения)'} не поддерживается: {path.name}. "
            f"Поддерживаются: {known}"
        )


def _aspose() -> Any:
    """Lazy import: the .NET runtime behind Aspose starts only when a file needs it."""
    try:
        import aspose.cells as cells
    except ImportError as exc:
        raise ConversionError(
            "Для конвертации нужен Aspose.Cells: pip install aspose-cells-python"
        ) from exc
    return cells


def convert_to_xlsx(source: Path, target: Path) -> list[str]:
    """Re-saves `source` as .xlsx at `target`. Returns the sheet names of the source.

    Formulas are not recalculated here: the values cached in the file are the
    ones Excel showed, and they are what the pipeline reads.
    """
    cells = _aspose()
    try:
        workbook = cells.Workbook(str(source))
        names = [sheet.name for sheet in workbook.worksheets]
        workbook.save(str(target), cells.SaveFormat.XLSX)
    except RuntimeError as exc:  # Aspose reports its .NET errors as RuntimeError
        raise ConversionError(f"Aspose.Cells не смог сконвертировать {source.name}: {exc}") from exc
    return names


@contextmanager
def prepared_workbook(path: Path) -> Iterator[PreparedWorkbook]:
    """The workbook as .xlsx for the duration of the `with` block.

    An .xlsx is handed on as it is. Any other supported format is converted into
    a temporary folder that is removed when the block ends — on an error too.
    """
    path = Path(path)
    check_input(path)
    if path.suffix.lower() == NATIVE_SUFFIX:
        yield PreparedWorkbook(path)
        return

    with tempfile.TemporaryDirectory(prefix="langfuse_parser_") as folder:
        target = Path(folder) / f"{path.stem}{NATIVE_SUFFIX}"
        sheets = convert_to_xlsx(path, target)
        logger.info(
            "[%s] сконвертирован через Aspose.Cells во временный .xlsx, листов: %d",
            path.name, len(sheets),
        )
        yield PreparedWorkbook(target, tuple(sheets))
