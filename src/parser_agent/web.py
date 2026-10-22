"""The web interface: a catalog browser over the finished database.

Everything the pipeline produced comes together here — the tables of the SQLite
file, the passports next to them and the points in Qdrant:

  * the list of tables with their descriptions and sizes;
  * one table in full: the columns, their types, samples and value domains;
  * the chat: every question runs the "chat" process of the flow — a crew in
    which a coordinator delegates the table search and the SQL to an analyst and
    a chart to a visualization specialist, and answers by the rows the database
    returned. When it cannot tell which table is meant, it asks.

The page itself is a single static file, `web/index.html`; this module only
serves the data to it. Nothing is served without signing in first
(`web/login.html`, accounts in auth.py); each chat turn is traced to Langfuse
under the signed-in user and the conversation it belongs to.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import asyncio
import json
import re
import time
from contextlib import nullcontext
from datetime import date, datetime, timezone

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse

from . import tracing
from .auth import COOKIE_NAME, DEFAULT_AUTH_DB, SESSION_SECONDS, AuthError, AuthStore
from .chats import DEFAULT_CHATS_DB, ChatStore, make_title
from .catalog import connect_existing, list_tables, quote_identifier
from .crew import progress
from .crew.chat_crew import ChatRun
from .indexer import DEFAULT_COLLECTION, DEFAULT_EMBED_MODEL, DEFAULT_URL, catalog_files
from .outputs import DEFAULT_OUTPUT_DIR
from .routing import DEFAULT_ROUTES, Task
from .search import read_catalogs

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "web"

# Everything but these needs a signed-in user. An allow-list, not a deny-list:
# a page or an endpoint added later is closed until it is put here on purpose.
PUBLIC_PATHS = frozenset({"/login", "/api/login", "/api/logout"})

# A chat conversation id from the page: it becomes the Langfuse session id,
# which must be US-ASCII.
_CONVERSATION_RE = re.compile(r"[A-Za-z0-9-]{8,64}")

# A chart id: 32 hex characters, nothing that could walk out of the folder.
_CHART_RE = re.compile(r"[0-9a-f]{32}")

DEFAULT_CHARTS_DIR = DEFAULT_OUTPUT_DIR / "charts"

# A chart that no saved answer shows (its turn failed to save) goes after a day.
UNSAVED_CHART_SECONDS = 24 * 3600

# A page that asks for this gets the steps of the answer while it is being made,
# one JSON object per line, and the answer itself as the last line.
STREAM_TYPE = "application/x-ndjson"

# The models of the chat crew when the caller does not pass them.
DEFAULT_CREW_MODELS = {
    "orchestrator": DEFAULT_ROUTES[Task.ORCHESTRATOR],
    "sql": DEFAULT_ROUTES[Task.SQL],
    "viz": DEFAULT_ROUTES[Task.VIZ],
}





def _connect(db_path: Path) -> sqlite3.Connection:
    """Подключение к существующей базе: пустышку по опечатке не создаём."""
    connection = connect_existing(db_path)
    connection.text_factory = str
    return connection


def prune_charts(charts_dir: Path, in_use: set[str], *, now: float | None = None) -> None:
    """Deletes the charts that no saved answer shows, once they are a day old.
    A chart a conversation shows lives as long as the conversation does."""
    now = time.time() if now is None else now
    for path in charts_dir.glob("*.png"):
        try:
            if path.stem not in in_use and now - path.stat().st_mtime > UNSAVED_CHART_SECONDS:
                path.unlink()
        except OSError:
            pass


def _as_list(tables: frozenset[str] | None) -> list[str] | None:
    return None if tables is None else sorted(tables)


def database_updated_at(db_path: Path) -> str | None:
    """When the tables were last written, in UTC ISO-8601; None without a database.

    SQLite keeps no time per table, but the file changes only when `convert`
    rewrites the tables: the chat opens it read-only. So the file's own time is
    the time of the last update.
    """
    if not db_path.exists():
        return None
    return datetime.fromtimestamp(db_path.stat().st_mtime, tz=timezone.utc).isoformat()


def table_sizes(db_path: Path) -> dict[str, dict[str, int]]:
    """{table: {rows, columns}} — read from the database itself."""
    if not db_path.exists():
        return {}
    connection = _connect(db_path)
    try:
        sizes: dict[str, dict[str, int]] = {}
        for name in list_tables(connection):
            rows = connection.execute(
                f"SELECT COUNT(*) FROM {quote_identifier(name)}"
            ).fetchone()[0]
            columns = len(
                connection.execute(f"PRAGMA table_info({quote_identifier(name)})").fetchall()
            )
            sizes[name] = {"rows": rows, "columns": columns}
        return sizes
    finally:
        connection.close()














def create_app(
    db_path: Path,
    catalog_dir: Path,
    *,
    qdrant_url: str = DEFAULT_URL,
    collection: str = DEFAULT_COLLECTION,
    embed_model: str = DEFAULT_EMBED_MODEL,
    crew_models: dict[str, str] | None = None,
    charts_dir: Path = DEFAULT_CHARTS_DIR,
    auth_db: Path = DEFAULT_AUTH_DB,
    chats_db: Path = DEFAULT_CHATS_DB,
) -> FastAPI:
    """The application. The paths are read on every request — no restart needed.

    Every page and endpoint needs a signed-in user (auth.py); the accounts live
    in auth_db and are made from the terminal. The conversations are saved in
    chats_db (chats.py), each visible to its owner only.
    """
    tracing.init()   # every chat turn becomes a trace, when Langfuse is configured
    store = AuthStore(auth_db)
    chats = ChatStore(chats_db)
    app = FastAPI(title="Parser Agent", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def require_login(request: Request, call_next: Any) -> Any:
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)
        user = store.user_for(request.cookies.get(COOKIE_NAME))
        if user is None:
            if request.url.path.startswith("/api/"):
                return JSONResponse({"detail": "Not signed in"}, status_code=401)
            return RedirectResponse("/login", status_code=303)
        request.state.user = user
        return await call_next(request)

    @app.get("/login")
    def login_page(request: Request) -> Any:
        if store.user_for(request.cookies.get(COOKIE_NAME)):
            return RedirectResponse("/", status_code=303)
        return FileResponse(STATIC_DIR / "login.html")

    @app.post("/api/login")
    def login(request: Request, payload: dict = Body(...)) -> JSONResponse:
        username = str(payload.get("username") or "").strip()
        try:
            token = store.login(username, str(payload.get("password") or ""))
        except AuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        response = JSONResponse({"user": username})
        response.set_cookie(
            COOKIE_NAME, token, max_age=SESSION_SECONDS, path="/",
            httponly=True,        # not readable from JavaScript
            samesite="strict",    # not sent with requests started by other sites
            secure=request.url.scheme == "https",
        )
        return response

    @app.post("/api/logout")
    def logout(request: Request) -> JSONResponse:
        store.logout(request.cookies.get(COOKIE_NAME))
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE_NAME, path="/")
        return response

    @app.get("/api/me")
    def me(request: Request) -> JSONResponse:
        return JSONResponse({"user": request.state.user})

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/overview")
    def overview(request: Request) -> JSONResponse:
        catalogs = read_catalogs(catalog_dir)
        sizes = table_sizes(db_path)
        # Only the tables this account may read are counted and listed.
        allowed = store.allowed_tables(request.state.user)
        if allowed is not None:
            catalogs = {name: c for name, c in catalogs.items() if name in allowed}
            sizes = {name: s for name, s in sizes.items() if name in allowed}
        tables = []
        for name, catalog in sorted(catalogs.items()):
            size = sizes.get(name, {})
            tables.append(
                {
                    "table_name": name,
                    "schema_name": catalog.get("schema_name", ""),
                    "table_description": catalog.get("table_description", ""),
                    "columns": len(catalog.get("columns") or []),
                    "rows": size.get("rows"),
                    "in_database": name in sizes,
                    "with_values": sum(
                        1
                        for column in catalog.get("columns") or []
                        if column.get("allowed_data")
                    ),
                }
            )
        # Tables that are in the database but have no passport yet.
        for name in sorted(set(sizes) - set(catalogs)):
            tables.append(
                {
                    "table_name": name,
                    "schema_name": "",
                    "table_description": "",
                    "columns": sizes[name]["columns"],
                    "rows": sizes[name]["rows"],
                    "in_database": True,
                    "with_values": 0,
                    "no_catalog": True,
                }
            )
        return JSONResponse(
            {
                "tables": tables,
                "totals": {
                    "tables": len(tables),
                    "columns": sum(t["columns"] for t in tables),
                    "rows": sum(t["rows"] or 0 for t in tables),
                    "catalogs": len(catalogs),
                },
                "updated_at": database_updated_at(db_path),
                "sources": {
                    "database": str(db_path),
                    "catalog_dir": str(catalog_dir),
                    "qdrant_url": qdrant_url,
                    "collection": collection,
                },
            }
        )

    # Turns still being answered for a page that has gone: kept referenced, so
    # they finish and are saved — the question shows up in the history anyway.
    turns_in_flight: set[asyncio.Task] = set()

    @app.post("/api/chat")
    async def chat(request: Request, payload: dict = Body(...)) -> Any:
        """One turn of the chat: the "chat" process of the flow.

        The Jev tree answers the lookups it is sure of; the rest goes to the
        Jev loop, where Jev sends the question to the analyst (search and SQL)
        and the visualization specialist, and the answer analyst writes the
        answer. The answer carries the SQL, the rows and the chart the tools
        actually produced.

        With `Accept: application/x-ndjson` the page gets the steps first, as
        the agents go through them — {"type": "progress", "steps": [...]} per
        line — and then {"type": "result", "result": <the answer>}. Without it,
        the answer alone, as one JSON object.
        """
        from .flow import run_chat   # CrewAI loads with the first question, not with the app

        message = str(payload.get("message") or "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Empty message")
        history = [
            {"role": str(turn.get("role", "")), "content": str(turn.get("content", ""))}
            for turn in (payload.get("history") or [])
            if isinstance(turn, dict)
        ]

        # The user comes from the session cookie, not from the page: it cannot
        # be claimed. The conversation id comes from the page; one per chat.
        user = request.state.user
        conversation = str(payload.get("session_id") or "")
        if not _CONVERSATION_RE.fullmatch(conversation):
            conversation = f"chat-{user}-{date.today().isoformat()}"
        owner = chats.owner(conversation)
        if owner is not None and owner != user:
            raise HTTPException(status_code=403, detail="This conversation belongs to another account")

        # The query behind the previous answer: a follow-up such as "draw it" is
        # charted from the same rows, read again from the database.
        previous = chats.last_query(user, conversation) or {}
        run = ChatRun(
            question=message, history=history,
            previous_sql=previous.get("sql"), previous_table=previous.get("table"),
            db_path=str(db_path), catalog_dir=str(catalog_dir), charts_dir=str(charts_dir),
            qdrant_url=qdrant_url, collection=collection, embed_model=embed_model,
            models=crew_models or DEFAULT_CREW_MODELS,
            allowed_tables=_as_list(store.allowed_tables(user)),
        )

        async def answer(listener: progress.Listener | None) -> dict[str, Any]:
            steps = progress.reporting(progress.Progress(listener)) if listener else nullcontext()
            with steps, tracing.trace(
                "chat", title=f"chat: {user}", input={"question": message},
                tags=["chat", "crew"], user_id=user, session_id=conversation,
            ) as root:
                result = await run_chat(run)
                if root is not None:
                    root.update(
                        output={key: result.get(key) for key in ("type", "text", "table", "sql", "chart")},
                        level="ERROR" if result.get("type") == "error" else None,
                    )
            tracing.flush()
            try:
                chats.save_turn(user, conversation, message, result)
            except Exception as exc:  # the answer still reaches the user; only the history misses it
                logger.warning("чат не сохранён: %s", exc)
            prune_charts(charts_dir, chats.charts_in_use())
            result["conversation"] = {"id": conversation, "title": make_title(message)}
            return result

        if STREAM_TYPE not in request.headers.get("accept", ""):
            return JSONResponse(await answer(None))

        # The agents run in CrewAI's threads: their steps come back to this
        # loop through call_soon_threadsafe, the answer through the task's end.
        loop = asyncio.get_running_loop()
        lines: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def on_steps(steps: list[progress.Step]) -> None:
            loop.call_soon_threadsafe(lines.put_nowait, {"type": "progress", "steps": steps})

        def on_done(task: asyncio.Task) -> None:
            turns_in_flight.discard(task)
            if task.cancelled():
                result: dict[str, Any] = {"type": "error", "text": "The answer was cancelled"}
            elif task.exception() is not None:
                logger.warning("chat turn failed: %s", task.exception())
                result = {"type": "error", "text": f"The assistant could not answer: {task.exception()}"}
            else:
                result = task.result()
            lines.put_nowait({"type": "result", "result": result})

        turn = asyncio.create_task(answer(on_steps))
        turns_in_flight.add(turn)
        turn.add_done_callback(on_done)

        async def stream() -> Any:
            while True:
                line = await lines.get()
                yield json.dumps(line, ensure_ascii=False, default=str) + "\n"
                if line["type"] == "result":
                    return

        return StreamingResponse(
            stream(), media_type=STREAM_TYPE,
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    # --- saved conversations ------------------------------------------------------

    @app.get("/api/chats")
    def list_chats(request: Request) -> JSONResponse:
        return JSONResponse({"chats": chats.list_conversations(request.state.user)})

    @app.get("/api/chats/{conversation_id}")
    def open_chat(request: Request, conversation_id: str) -> JSONResponse:
        chat = chats.get(request.state.user, conversation_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="No such conversation")
        return JSONResponse(chat)

    @app.patch("/api/chats/{conversation_id}")
    def rename_chat(request: Request, conversation_id: str, payload: dict = Body(...)) -> JSONResponse:
        if not chats.rename(request.state.user, conversation_id, str(payload.get("title") or "")):
            raise HTTPException(status_code=404, detail="No such conversation")
        return JSONResponse({"ok": True})

    @app.delete("/api/chats/{conversation_id}")
    def delete_chat(request: Request, conversation_id: str) -> JSONResponse:
        charts = chats.delete(request.state.user, conversation_id)
        if charts is None:
            raise HTTPException(status_code=404, detail="No such conversation")
        for chart_id in charts:
            if _CHART_RE.fullmatch(chart_id):
                (charts_dir / f"{chart_id}.png").unlink(missing_ok=True)
        return JSONResponse({"ok": True})

    @app.get("/api/charts/{chart_id}")
    def chart(request: Request, chart_id: str) -> FileResponse:
        # A chart is shown to the owner of the conversation it belongs to only.
        path = charts_dir / f"{chart_id}.png"
        if (not _CHART_RE.fullmatch(chart_id) or not path.is_file()
                or chats.chart_owner(chart_id) != request.state.user):
            raise HTTPException(status_code=404, detail="No such chart")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/status")
    def status() -> JSONResponse:
        """What the page shows in the sidebar: are the sources alive."""
        state = {
            "database": db_path.exists(),
            "catalogs": len(catalog_files(catalog_dir)),
            "qdrant": False,
            "points": None,
        }
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=qdrant_url)
            state["points"] = client.count(collection).count
            state["qdrant"] = True
        except Exception as exc:
            logger.debug("Qdrant недоступен: %s", exc)
        return JSONResponse(state)

    return app
