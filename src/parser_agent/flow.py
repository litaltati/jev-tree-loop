"""Parser Agent as a CrewAI Flow: two independent processes behind one router.

    begin ─▶ which_process ─┬─ "parse" ─▶ prepare → compress → describe → load → passports → index
                            └─ "chat"  ─▶ answer: the Jev tree, then the Jev loop, for one question

A run is one or the other, chosen by `mode` in the inputs. The CLI runs "parse"
for a workbook; the web chat runs "chat" for every question a user asks. The
two share nothing but the database the first one writes and the second reads.

The parse process. The order of its stages is set by the code, not by a model:
no agent decides what to call next. The workbook is converted and every sheet is
compressed before the first request leaves for a model, and the model is shown
the compressed text only.

    prepare   : format check -> Aspose (.xlsb -> temporary .xlsx) -> workbook in memory
    compress  : SheetCompressor and the header chains of every sheet   (no LLM)
    describe  : the schema of every sheet from its compressed text     (LLM)
    load      : every sheet into SQLite by its schema                  (no LLM)
    passports : the passport of every table                            (LLM)
    index     : the passports into Qdrant, when asked for              (embeddings)

The stages themselves are the modules next to this one; the Flow only runs
them. Each step finishes for all sheets before the next one starts.

A sheet that fails at describe, load or passports is marked with its error and
left out of the steps after it; the other sheets go on. One sheet the model
could not read must not cost the whole workbook. The failures are listed in
the state (`failed_sheets`), and the callers turn them into an exit code.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from crewai.flow.flow import Flow, listen, router, start
from openpyxl.utils import get_column_letter
from pydantic import BaseModel, Field, PrivateAttr

from . import tracing
from .catalog import DEFAULT_SCHEMA_NAME, build_catalog, catalog_json
from .crew.chat_crew import ChatRun
from .jev_loop.loop import answer_with_loop
from .jev_tree.chat import answer_with_tree
from .compressor import (
    MIN_NON_EMPTY_ROWS,
    compress_sheet_with_stats,
    non_empty_row_count,
    prepared_workbook,
)
from .describer import describe_sheet
from .extractor import extract_to_sqlite
from .indexer import DEFAULT_COLLECTION, DEFAULT_EMBED_MODEL, DEFAULT_URL, index_catalogs
from .layout import detect_header_layout, format_header_stacks, header_stacks
from .models import SheetSchema, sanitize_table_name
from .outputs import DEFAULT_OUTPUT_DIR, pretty_json, write_output

logger = logging.getLogger(__name__)


class SheetRun(BaseModel):
    """One sheet on its way through the flow."""

    sheet_name: str
    compressed: str = ""                   # the text the model is shown
    header_block: str | None = None        # the glued header chains, when the layout is known
    header_columns: list[str] = Field(default_factory=list)  # the letters the chains cover
    sheet_schema: SheetSchema | None = None
    chains: dict[str, str] = Field(default_factory=dict)  # {column: header chain}
    loaded: bool = False                   # the table is in SQLite
    rows: int = 0                          # rows loaded into SQLite
    error: str | None = None               # "<step>: <message>" — the sheet stopped there

    @property
    def active(self) -> bool:
        """Whether the sheet still goes on to the next step."""
        return self.error is None


class ParserAgentState(BaseModel):
    """The inputs of one run and what the steps have produced so far."""

    mode: str = "parse"                    # "parse" | "chat" — which process runs
    chat: ChatRun | None = None            # the question, for mode "chat"

    input_path: str = ""
    output_db: str = "output.db"
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    dry_run: bool = False                  # compression only: no LLM, no database
    dump_dir: str | None = None            # where to save the compressed text of every sheet
    skip_aggregation: bool = False
    expand_hierarchy: bool = True
    compose_names: bool = False
    detect_layout: bool = True
    skip_empty_columns: bool = True
    skip_text_only_rows: bool = True
    make_catalog: bool = True
    schema_name: str = DEFAULT_SCHEMA_NAME
    describe_model: str | None = None      # None — the model from routing.py
    catalog_model: str | None = None
    index: bool = False                    # load the passports into Qdrant at the end
    qdrant_url: str = DEFAULT_URL
    collection: str = DEFAULT_COLLECTION
    embed_model: str = DEFAULT_EMBED_MODEL

    sheets: list[SheetRun] = Field(default_factory=list)
    total_rows: int = 0
    indexed_points: int | None = None

    @property
    def failed_sheets(self) -> list[SheetRun]:
        return [sheet for sheet in self.sheets if sheet.error is not None]

    @property
    def loaded_sheets(self) -> list[SheetRun]:
        return [sheet for sheet in self.sheets if sheet.loaded]


class ParserAgentFlow(Flow[ParserAgentState]):
    """Excel -> SQLite and passports, stage by stage, in a fixed order."""

    # Nothing here is remembered between runs. Without this the Flow would build
    # its default memory — a LanceDB store with an OpenAI embedder — for nothing.
    # CrewAI's own internal flows opt out the same way.
    _skip_auto_memory: bool = True

    _client: Any = PrivateAttr(default=None)
    _echo: Callable[[str], None] = PrivateAttr(default=print)
    _workbook: Any = PrivateAttr(default=None)

    def __init__(
        self,
        *,
        client: Any | None = None,
        echo: Callable[[str], None] = print,
        **data: Any,
    ) -> None:
        """client — an async OpenAI-compatible client for the LLM steps (None — the
        default one); echo — where the progress lines go."""
        super().__init__(**data)
        self._client = client
        self._echo = echo

    @start()
    def begin(self) -> None:
        """Nothing to do here: the router right after it picks the process."""

    @router(begin)
    def which_process(self) -> str:
        return "chat" if self.state.mode == "chat" else "parse"

    # --- the chat process --------------------------------------------------------

    @listen("chat")
    @tracing.observe("crew")
    async def answer(self) -> None:
        """One question: the Jev tree first (jev_tree/chat.py), then the Jev loop.

        Stage 1 of the tree settles what the question needs: the tree for one
        line, the SQL agent for more, no data at all. Whatever the tree found —
        its rows, its notes, the first step it already knows — goes into the
        loop's one state (jev_loop/); there Jev picks the visualizer when a chart
        is wanted, and the answer analyst writes every answer. The tree runs in
        a thread: its calls to Jev are blocking, and the context goes with it, so the steps and the Langfuse
        spans stay with this question.
        """
        run = self.state.chat
        if run is None:
            raise ValueError("mode 'chat' needs the question in `chat`")
        found = await asyncio.to_thread(answer_with_tree, run)
        run.answer = await answer_with_loop(run, found)
        tracing.update(output={k: run.answer.get(k) for k in ("type", "engine", "text", "table", "sql", "chart")})

    # --- the parse process -------------------------------------------------------

    @listen("parse")
    @tracing.observe("prepare")
    def prepare(self) -> None:
        """The workbook as .xlsx in memory, and the sheets worth parsing."""
        state = self.state
        with prepared_workbook(Path(state.input_path)) as source:
            self._workbook = source.load()
            names = source.sheet_names(self._workbook)
        # A temporary copy is already gone here: openpyxl has read the whole file.
        for name in names:
            count = non_empty_row_count(self._workbook[name])
            if count < MIN_NON_EMPTY_ROWS:
                logger.info("[%s] пропущен: непустых строк %d", name, count)
                continue
            state.sheets.append(SheetRun(sheet_name=name))
        tracing.update(output={"sheets": [sheet.sheet_name for sheet in state.sheets]})

    @listen(prepare)
    @tracing.observe("compress")
    def compress(self) -> None:
        """Every sheet compressed — before any model has seen a single cell."""
        state = self.state
        summary: dict[str, str] = {}
        for sheet in state.sheets:
            ws = self._workbook[sheet.sheet_name]
            sheet.compressed, stats = compress_sheet_with_stats(
                ws, skip_aggregation=state.skip_aggregation
            )
            summary[sheet.sheet_name] = (
                f"{stats.original_tokens} -> {stats.compressed_tokens} tokens "
                f"({stats.ratio:.2f}x)"
            )
            self._echo(
                f"[{sheet.sheet_name}] сжатие: {stats.original_tokens} -> "
                f"{stats.compressed_tokens} токенов (ratio {stats.ratio:.2f}x)"
            )
            if state.dump_dir is not None:
                dump_dir = Path(state.dump_dir)
                dump_dir.mkdir(parents=True, exist_ok=True)
                dump_path = dump_dir / f"{sanitize_table_name(sheet.sheet_name)}.txt"
                dump_path.write_text(sheet.compressed, encoding="utf-8")
                logger.debug("[%s] сжатый текст сохранён: %s", sheet.sheet_name, dump_path)
            if not state.dry_run:
                # The model is shown the header cells; the names are its job.
                layout = detect_header_layout(ws)
                if layout.detected:
                    stacks = header_stacks(ws, layout)
                    sheet.header_block = format_header_stacks(stacks)
                    sheet.header_columns = [get_column_letter(col) for col in stacks]
        tracing.update(output=summary)

    @listen(compress)
    @tracing.observe("describe")
    async def describe(self) -> None:
        """The schema of every sheet from its compressed text: the first LLM call of a run."""
        state = self.state
        if state.dry_run:
            return
        out_dir = Path(state.output_dir)
        for sheet in state.sheets:
            name = sheet.sheet_name
            with tracing.span(f"sheet {name}"):
                try:
                    # The model answer is saved as is — before parsing and sanitization.
                    schema = await describe_sheet(
                        sheet.compressed, name,
                        model=state.describe_model, client=self._client,
                        header_block=sheet.header_block,
                        on_raw_response=lambda raw, name=name: write_output(
                            out_dir, name, ".response.json", pretty_json(raw)
                        ),
                    )
                except Exception as exc:  # an API error or an answer that failed its retry
                    self._fail(sheet, "describe", exc)
                    continue
                # The model is told to describe every column of the header list;
                # whether it did is checked here, not trusted.
                described = {c.original_col_letter for c in schema.columns}
                skipped = [letter for letter in sheet.header_columns if letter not in described]
                tracing.update(
                    output={
                        "table": schema.table_name, "columns": [c.name for c in schema.columns],
                        "rows": f"{schema.data_start_row}-{schema.data_end_row}",
                        "skipped_columns": skipped,
                    },
                    level="WARNING" if skipped else None,
                )
            if skipped:
                shown = ", ".join(skipped[:15]) + (" …" if len(skipped) > 15 else "")
                self._echo(
                    f"[{name}] ВНИМАНИЕ: модель пропустила {len(skipped)} из "
                    f"{len(sheet.header_columns)} колонок, их не будет в таблице: {shown}"
                )
            saved = write_output(out_dir, name, ".schema.json", schema.model_dump_json(indent=2))
            if saved is not None:
                self._echo(f"[{name}] ответ модели: {saved.parent}/")
            logger.debug("[%s] схема: %s", name, schema.model_dump_json())
            self._echo(
                f"[{name}] схема: таблица {schema.table_name}, колонок "
                f"{len(schema.columns)}, строки {schema.data_start_row}-{schema.data_end_row}"
            )
            sheet.sheet_schema = schema

    @listen(describe)
    @tracing.observe("load")
    def load(self) -> None:
        """Every sheet into SQLite by its schema."""
        state = self.state
        if state.dry_run:
            return
        Path(state.output_db).parent.mkdir(parents=True, exist_ok=True)
        for sheet in state.sheets:
            if not sheet.active:
                continue
            with tracing.span(f"sheet {sheet.sheet_name}"):
                try:
                    # The chains are collected on the way out: only here, with the sheet
                    # open, is it known which chain became which column.
                    sheet.rows = extract_to_sqlite(
                        state.input_path, sheet.sheet_schema, state.output_db,
                        ws=self._workbook[sheet.sheet_name],   # the sheet is in memory — no second read
                        drop_existing=True, expand_hierarchy=state.expand_hierarchy,
                        compose_names=state.compose_names, detect_layout=state.detect_layout,
                        skip_empty_columns=state.skip_empty_columns,
                        skip_text_only_rows=state.skip_text_only_rows,
                        on_columns=sheet.chains.update,
                    )
                except Exception as exc:  # a schema the sheet does not fit, a database error
                    self._fail(sheet, "load", exc)
                    continue
                tracing.update(output={"rows": sheet.rows})
            sheet.loaded = True
            state.total_rows += sheet.rows
            self._echo(f"[{sheet.sheet_name}] загружено строк: {sheet.rows}")

    @listen(load)
    @tracing.observe("passports")
    async def passports(self) -> None:
        """The passport of every table.

        Only now is the whole database there: a table is described by what tells
        it apart from its neighbours.
        """
        state = self.state
        if state.dry_run or not state.make_catalog:
            return
        for sheet in state.sheets:
            if not sheet.active:
                continue
            schema = sheet.sheet_schema
            with tracing.span(f"sheet {sheet.sheet_name}"):
                try:
                    passport = await build_catalog(
                        state.output_db, schema.table_name,
                        schema_name=state.schema_name, sheet_name=sheet.sheet_name,
                        chains=sheet.chains, notes=schema.notes,
                        model=state.catalog_model, client=self._client,
                    )
                except Exception as exc:  # the table stays in SQLite, only without a passport
                    self._fail(sheet, "passports", exc)
                    continue
                tracing.update(output={"table_description": passport.get("table_description", "")})
            saved = write_output(
                Path(state.output_dir), sheet.sheet_name, ".catalog.json", catalog_json(passport)
            )
            if saved is not None:
                self._echo(f"[{sheet.sheet_name}] паспорт таблицы: {saved}")

    @listen(passports)
    @tracing.observe("index")
    def index(self) -> None:
        """The passports into Qdrant — only when the run asked for it."""
        state = self.state
        if state.dry_run or not state.index:
            return
        points = index_catalogs(
            Path(state.output_dir), collection=state.collection,
            url=state.qdrant_url, embed_model=state.embed_model,
        )
        state.indexed_points = len(points)
        self._echo(f"паспорта -> Qdrant: точек {len(points)} в {state.collection}")

    def _fail(self, sheet: SheetRun, step: str, exc: Exception) -> None:
        """Marks the sheet as stopped at `step`; the other sheets go on."""
        sheet.error = f"{step}: {exc}"
        logger.debug("[%s] %s failed", sheet.sheet_name, step, exc_info=exc)
        tracing.update(level="ERROR", status_message=sheet.error)   # the sheet's own span
        self._echo(f"[{sheet.sheet_name}] ОШИБКА ({step}), лист пропущен: {exc}")

    def release_workbook(self) -> None:
        """Closes the workbook the steps read from."""
        if self._workbook is not None:
            self._workbook.close()
            self._workbook = None


def decline_crewai_cloud_traces() -> None:
    """Never ask about CrewAI's cloud traces, and never wait for the answer.

    After the first run in a project folder CrewAI asks in the terminal whether
    to upload the execution traces to its cloud, and waits 20 s for a reply. It
    reads "is this the first run" once per process, so in the web server the
    question came back after every answer — and every answer waited for it.

    The traces of this project go to its own Langfuse; nothing goes to CrewAI.
    The refusal is saved, as answering "n" would, and the listener of this
    process is told so as well. Without consent CrewAI sends nothing.
    """
    from crewai.events.listeners.tracing.trace_listener import TraceCollectionListener
    from crewai.events.listeners.tracing.utils import mark_first_execution_done

    mark_first_execution_done(user_consented=False)
    TraceCollectionListener().first_time_handler.is_first_time = False


async def run_chat(run: ChatRun) -> dict[str, Any]:
    """The chat process for one question. Returns what the web page renders.

    The CrewAI panels are left out: this runs inside the web server, once per
    question, and its console is not where anybody reads them.
    """
    decline_crewai_cloud_traces()
    flow = ParserAgentFlow(suppress_flow_events=True)
    await flow.kickoff_async(inputs={"mode": "chat", "chat": run.model_dump()})
    return flow.state.chat.answer or {"type": "error", "text": "No answer was produced"}


def run_parser_agent(
    inputs: dict[str, Any],
    *,
    client: Any | None = None,
    echo: Callable[[str], None] = print,
) -> ParserAgentState:
    """Runs the flow once, for one workbook. Returns the final state.

    With Langfuse on, the run is one trace: the steps and every model call nest
    under its root, and the outcome goes into the trace as scores.
    """
    tracing.init()
    decline_crewai_cloud_traces()
    flow = ParserAgentFlow(client=client, echo=echo)
    try:
        with tracing.workbook_trace(inputs) as root:
            try:
                flow.kickoff(inputs=inputs)
            finally:
                flow.release_workbook()
            tracing.finish_workbook(root, flow.state)
    finally:
        tracing.flush()
    return flow.state
