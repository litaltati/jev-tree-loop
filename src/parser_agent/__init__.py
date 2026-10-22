"""Parser Agent: Excel -> SheetCompressor -> LLM schema -> SQLite.

The stages, run in this order by the CrewAI Flow in flow.py:
  0. convert   — a non-.xlsx workbook is re-saved as .xlsx by Aspose.Cells
                 (compressor/converter.py)
  1. compress  — LLM-free compression of the sheet (compressor/)
  2. describe  — recognition of the sheet schema by a model (describer.py)
  3. extract   — deterministic loading into SQLite (extractor.py)
  4. catalog   — the passport of every table (catalog.py)
  5. index     — the passports into Qdrant, when asked for (indexer.py)

Rule: 1 Excel sheet = 1 SQLite table.

The flow is not imported here: CrewAI is heavy to import, and the commands
that do not run the flow (serve, models, prompt) do not need it.
"""

from .compressor import compress_sheet, compress_sheet_with_stats
from .describer import describe_sheet, describe_sheet_sync
from .extractor import extract_to_sqlite
from .models import ColumnSchema, SheetSchema, sanitize_table_name

__version__ = "0.1.0"

__all__ = [
    "ColumnSchema",
    "SheetSchema",
    "compress_sheet",
    "compress_sheet_with_stats",
    "describe_sheet",
    "describe_sheet_sync",
    "extract_to_sqlite",
    "sanitize_table_name",
    "__version__",
]
