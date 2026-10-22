"""SheetCompressor: compression of an Excel sheet into compact text without an LLM.

Before it, the converter brings a non-.xlsx workbook to .xlsx (module 0).
"""

from .aggregator import aggregate, detect_type, nfs_category
from .anchors import CellInfo, CompactSheet, build_compact_sheet
from .converter import (
    CONVERTIBLE_SUFFIXES,
    NATIVE_SUFFIX,
    ConversionError,
    PreparedWorkbook,
    prepared_workbook,
)
from .inverted import build_inverted_index, fold_addresses, to_inverted_json
from .pipeline import (
    MIN_NON_EMPTY_ROWS,
    CompressionStats,
    compress_sheet,
    compress_sheet_with_stats,
    estimate_tokens,
    non_empty_row_count,
    vanilla_encoding,
)

__all__ = [
    "CONVERTIBLE_SUFFIXES",
    "MIN_NON_EMPTY_ROWS",
    "NATIVE_SUFFIX",
    "CellInfo",
    "CompactSheet",
    "CompressionStats",
    "ConversionError",
    "PreparedWorkbook",
    "aggregate",
    "build_compact_sheet",
    "build_inverted_index",
    "compress_sheet",
    "compress_sheet_with_stats",
    "detect_type",
    "estimate_tokens",
    "fold_addresses",
    "nfs_category",
    "non_empty_row_count",
    "prepared_workbook",
    "to_inverted_json",
    "vanilla_encoding",
]
