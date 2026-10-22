"""Where the files of a run go: the model answers, the schemas, the passports."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import sanitize_table_name

logger = logging.getLogger(__name__)

# The folder is created next to the launch directory so the result is visible
# and does not mix with the package sources.
DEFAULT_OUTPUT_DIR = Path("output")


def pretty_json(text: str) -> str:
    """Formats JSON for reading. Text that could not be parsed is left as is."""
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except ValueError:
        return text


def write_output(out_dir: Path, sheet_name: str, suffix: str, text: str) -> Path | None:
    """Writes a file into the output folder. None if the write failed."""
    path = out_dir / f"{sanitize_table_name(sheet_name)}{suffix}"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.warning("Не удалось записать %s: %s", path, exc)
        return None
    return path
