"""CrewAI entry points: `crewai run` and `crewai flow plot` land here.

`crewai run` passes no arguments, so it parses every workbook in data/;
`uv run kickoff <file> ...` parses the given files instead. Each workbook gets a
folder of its own in output/: two reports with the same sheet names would
otherwise overwrite each other's tables.

Nothing runs at import: `crewai run` imports this module first, to look for a
conversational flow in it.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from .compressor import CONVERTIBLE_SUFFIXES, NATIVE_SUFFIX
from .flow import ParserAgentFlow, run_parser_agent
from .outputs import DEFAULT_OUTPUT_DIR

DATA_DIR = Path("data")

SUPPORTED_SUFFIXES = frozenset({NATIVE_SUFFIX, *CONVERTIBLE_SUFFIXES})


def workbooks(folder: Path) -> list[Path]:
    """The workbooks of a folder the flow can read. Excel's lock files (~$...) are left out."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not path.name.startswith("~$")
    )


def kickoff() -> None:
    """Parses every workbook: output/<name>/<name>.db plus its passports next to it."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    # A key exported in the terminal takes priority over the file.
    load_dotenv(find_dotenv(usecwd=True), override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY не задан: добавьте его в .env в корне проекта.", file=sys.stderr)
        raise SystemExit(2)

    files = [Path(arg) for arg in sys.argv[1:]]
    if not files and DATA_DIR.is_dir():
        files = workbooks(DATA_DIR)
    if not files:
        print(f"В {DATA_DIR}/ нет Excel-файлов ({', '.join(sorted(SUPPORTED_SUFFIXES))}).",
              file=sys.stderr)
        raise SystemExit(2)

    failed: list[str] = []
    for path in files:
        folder = DEFAULT_OUTPUT_DIR / path.stem
        database = folder / f"{path.stem}.db"
        print(f"===== {path.name} -> {folder}/")
        try:
            state = run_parser_agent(
                {"input_path": str(path), "output_db": str(database), "output_dir": str(folder)}
            )
        except Exception as exc:  # one broken workbook must not stop the others
            print(f"[{path.name}] ошибка: {exc}", file=sys.stderr)
            failed.append(path.name)
            continue
        print(f"[{path.name}] готово: {database}, строк {state.total_rows}")
        for sheet in state.failed_sheets:
            print(f"[{path.name}] лист пропущен: {sheet.sheet_name} — {sheet.error}", file=sys.stderr)
            failed.append(f"{path.name}:{sheet.sheet_name}")

    if failed:
        print(f"Не разобраны: {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)


def plot() -> None:
    """The flow diagram as an HTML page: parser_agent_flow.html."""
    ParserAgentFlow().plot("parser_agent_flow.html")


if __name__ == "__main__":
    kickoff()
