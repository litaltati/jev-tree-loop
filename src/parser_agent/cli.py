"""Typer CLI: python -m parser_agent convert input.xlsx output.db"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import typer
from dotenv import find_dotenv, load_dotenv

from . import tracing
from .auth import DEFAULT_AUTH_DB, AuthError, AuthStore, check_username
from .catalog import DEFAULT_SCHEMA_NAME, build_catalog, catalog_json, list_tables
from .compressor import (
    MIN_NON_EMPTY_ROWS,
    ConversionError,
    compress_sheet_with_stats,
    estimate_tokens,
    non_empty_row_count,
    prepared_workbook,
)
from .describer import SYSTEM_PROMPT, build_user_prompt
from .indexer import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    DEFAULT_URL,
    index_catalogs,
)
from .layout import detect_header_layout, format_header_stacks, header_stacks
from .models import sanitize_table_name
from .outputs import DEFAULT_OUTPUT_DIR, write_output
from .routing import Task, build_router

logger = logging.getLogger("parser_agent")

app = typer.Typer(
    add_completion=False,
    help="Parser Agent: Excel -> SheetCompressor -> LLM-схема -> SQLite (1 лист = 1 таблица).",
)

INPUT_HELP = "Исходный .xlsx (.xlsb/.xls/.xlsm/.ods сначала конвертируются через Aspose.Cells)"


def _load_env(env_file: Path | None) -> str | None:
    """Loads variables from .env. Returns the path of the file that was used.

    Already defined environment variables take priority (override=False): a key
    exported in the terminal must not be silently replaced by the file.
    """
    if env_file is not None:
        if not env_file.exists():
            typer.echo(f"Файл {env_file} не найден.", err=True)
            raise typer.Exit(code=2)
        load_dotenv(env_file, override=False)
        return str(env_file)

    # Look for .env in the current directory and up the tree.
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found, override=False)
        return found
    return None


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _require_accounts(auth_db: Path) -> None:
    """The interface opens only behind a login, so there has to be someone to log in."""
    if not AuthStore(auth_db).has_users():
        typer.echo(
            f"В {auth_db} нет ни одного пользователя — интерфейс некому открыть.\n"
            "Создайте учётную запись: python3 -m parser_agent users add <имя>",
            err=True,
        )
        raise typer.Exit(code=2)


def _run_flow(inputs: dict[str, Any]) -> Any:
    """Runs Parser Agent — the CrewAI Flow — with its progress lines in the terminal.

    CrewAI is imported here and not at the top: the commands that do not run the
    flow should not pay for loading it.
    """
    from .flow import run_parser_agent

    return run_parser_agent(inputs, echo=typer.echo)


@app.callback()
def _root() -> None:
    """An empty callback: it makes Typer keep the `convert` subcommand name
    (with a single command Typer would otherwise collapse it into the app root)."""


@app.command()
def convert(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True,
                                      help=INPUT_HELP),
    output_db: Path = typer.Argument(..., dir_okay=False, help="Файл SQLite"),
    skip_aggregation: bool = typer.Option(
        False, "--skip-aggregation", help="Отключить модуль 3 (data-format-aware aggregation)"
    ),
    model: list[str] = typer.Option(
        None,
        "--model",
        help=(
            "Переопределить модель: `--model gpt-4o` для всех задач или "
            "`--model describe=gpt-4o` для одной. Флаг можно повторять."
        ),
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Подробный лог"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Только сжатие: без обращения к LLM и без записи в SQLite"
    ),
    dump_compressed: Path = typer.Option(
        None, "--dump-compressed", help="Каталог, куда сохранить сжатый текст каждого листа"
    ),
    env_file: Path = typer.Option(
        None, "--env-file", help="Путь к .env с OPENAI_API_KEY (по умолчанию ищется рядом)"
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        help="Папка для ответов модели: <лист>.response.json и <лист>.schema.json",
    ),
    flat: bool = typer.Option(
        False, "--flat", help="Не разворачивать иерархию меток строк в колонки уровней"
    ),
    compose_names: bool = typer.Option(
        False,
        "--compose-names",
        help="Склеивать имена колонок кодом через `__` вместо имён от модели",
    ),
    llm_layout: bool = typer.Option(
        False,
        "--llm-layout",
        help="Брать границы шапки от модели, не определять их по структуре листа",
    ),
    keep_empty_columns: bool = typer.Option(
        False,
        "--keep-empty-columns",
        help="Создавать колонки и для полностью пустых колонок Excel",
    ),
    keep_text_rows: bool = typer.Option(
        False,
        "--keep-text-rows",
        help="Сохранять строки, где нет ни одного числа (подзаголовки разделов)",
    ),
    no_catalog: bool = typer.Option(
        False,
        "--no-catalog",
        help="Не собирать паспорт таблицы (<лист>.catalog.json)",
    ),
    schema_name: str = typer.Option(
        DEFAULT_SCHEMA_NAME,
        "--schema-name",
        help="Значение поля schema_name в паспорте таблицы",
    ),
) -> None:
    """Converts every sheet of an .xlsx into SQLite tables (1 sheet = 1 table)."""
    _setup_logging(verbose)

    env_path = _load_env(env_file)
    if env_path:
        logger.debug("Переменные окружения загружены из %s", env_path)

    if not dry_run and not os.environ.get("OPENAI_API_KEY"):
        typer.echo(
            "OPENAI_API_KEY не задан.\n"
            "Создайте файл .env рядом с проектом со строкой OPENAI_API_KEY=sk-...\n"
            "или укажите путь через --env-file, либо экспортируйте переменную в терминале.\n"
            "Проверить сжатие без обращения к API можно с --dry-run.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        router = build_router(model or ())
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    for task, model_name, source in router.table():
        logger.debug("Модель для задачи %s: %s (%s)", task, model_name, source)

    try:
        state = _run_flow(
            {
                "input_path": str(input_path),
                "output_db": str(output_db),
                "output_dir": str(output_dir),
                "dry_run": dry_run,
                "dump_dir": str(dump_compressed) if dump_compressed is not None else None,
                "skip_aggregation": skip_aggregation,
                "expand_hierarchy": not flat,
                "compose_names": compose_names,
                "detect_layout": not llm_layout,
                "skip_empty_columns": not keep_empty_columns,
                "skip_text_only_rows": not keep_text_rows,
                "make_catalog": not no_catalog,
                "schema_name": schema_name,
                "describe_model": router.model_for(Task.DESCRIBE),
                "catalog_model": router.model_for(Task.CATALOG),
            }
        )
    except ConversionError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    if dry_run:
        typer.echo("Готово (--dry-run): схемы не запрашивались, база не записана.")
    else:
        typer.echo(f"Готово: {output_db}, всего строк {state.total_rows}")
    if _report_failures(state):
        raise typer.Exit(code=1)


def _report_failures(state: Any) -> bool:
    """Lists the sheets the flow had to skip. True if there were any."""
    for sheet in state.failed_sheets:
        typer.echo(f"Лист пропущен: {sheet.sheet_name} — {sheet.error}", err=True)
    return bool(state.failed_sheets)


@app.command()
def catalog(
    db_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Готовый файл SQLite"
    ),
    out: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--out", help="Папка для <таблица>.catalog.json"
    ),
    schema_name: str = typer.Option(
        DEFAULT_SCHEMA_NAME, "--schema-name", help="Значение поля schema_name"
    ),
    table: str = typer.Option(
        None, "--table", help="Только одна таблица (по умолчанию — все)"
    ),
    no_descriptions: bool = typer.Option(
        False,
        "--no-descriptions",
        help="Только факты из базы, без обращения к модели",
    ),
    model: list[str] = typer.Option(
        None, "--model", help="Переопределить модель (та же форма, что у convert)"
    ),
    env_file: Path = typer.Option(None, "--env-file", help="Путь к .env"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Подробный лог"),
) -> None:
    """Builds the passport of every table of a ready database.

    The same file `convert` writes at the end of a run. Here it is built for a
    database made earlier — the header chains of the source sheets are gone by
    then, so the descriptions come out of the column names and the data alone.
    """
    _setup_logging(verbose)
    _load_env(env_file)

    if not no_descriptions and not os.environ.get("OPENAI_API_KEY"):
        typer.echo(
            "OPENAI_API_KEY не задан. Соберите только факты из базы с "
            "--no-descriptions или укажите ключ через --env-file.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        router = build_router(model or ())
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    connection = sqlite3.connect(db_path)
    try:
        tables = list_tables(connection)
    finally:
        connection.close()

    if table is not None:
        if table not in tables:
            typer.echo(
                f"Таблица {table!r} не найдена. Доступные: {', '.join(tables) or '—'}",
                err=True,
            )
            raise typer.Exit(code=2)
        tables = [table]

    if not tables:
        typer.echo("В базе нет таблиц.", err=True)
        raise typer.Exit(code=1)

    tracing.init()
    for name in tables:
        with tracing.trace("catalog", title=f"catalog: {name}", input={"table": name},
                           tags=["catalog"]):
            passport = asyncio.run(
                build_catalog(
                    str(db_path), name,
                    schema_name=schema_name,
                    model=router.model_for(Task.CATALOG),
                    describe=not no_descriptions,
                )
            )
        saved = write_output(out, name, ".catalog.json", catalog_json(passport))
        if saved is not None:
            typer.echo(f"[{name}] паспорт таблицы: {saved}")
    tracing.flush()


@app.command()
def index(
    source: Path = typer.Argument(
        DEFAULT_OUTPUT_DIR,
        help="Папка с файлами <таблица>.catalog.json или один такой файл",
    ),
    url: str = typer.Option(DEFAULT_URL, "--url", help="Адрес Qdrant"),
    collection: str = typer.Option(
        DEFAULT_COLLECTION, "--collection", help="Коллекция Qdrant"
    ),
    embed_model: str = typer.Option(
        DEFAULT_EMBED_MODEL, "--embed-model", help="Модель эмбеддингов OpenAI"
    ),
    qdrant_api_key: str = typer.Option(
        None, "--qdrant-api-key", help="Ключ Qdrant, если сервер закрыт"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Показать текст каждой точки, ничего не считая и не записывая",
    ),
    env_file: Path = typer.Option(None, "--env-file", help="Путь к .env"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Подробный лог"),
) -> None:
    """Loads the table passports into Qdrant: one file — one point."""
    _setup_logging(verbose)
    _load_env(env_file)

    if not dry_run and not os.environ.get("OPENAI_API_KEY"):
        typer.echo(
            "OPENAI_API_KEY не задан — без него не посчитать эмбеддинги.\n"
            "Посмотреть, что уйдёт в Qdrant, можно с --dry-run.",
            err=True,
        )
        raise typer.Exit(code=2)

    if not dry_run:
        tracing.init()
    try:
        with tracing.trace("index", title=f"index: {source}", input={"source": str(source)},
                           tags=["index"]):
            points = index_catalogs(
                source,
                collection=collection,
                url=url,
                api_key=qdrant_api_key,
                embed_model=embed_model,
                dry_run=dry_run,
            )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    finally:
        tracing.flush()

    if dry_run:
        for point in points:
            typer.echo(f"----- {point['payload']['table_name']} ({point['id']}) -----")
            typer.echo(point["text"])
            typer.echo("")
        typer.echo(f"Готово (--dry-run): точек {len(points)}, ничего не записано.")
        return

    typer.echo(f"Готово: в коллекцию {collection} записано точек {len(points)} ({url})")


@app.command("all")
def run_all(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help=INPUT_HELP
    ),
    output_db: Path = typer.Argument(Path("output.db"), dir_okay=False, help="Файл SQLite"),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--output-dir", help="Папка для паспортов и ответов модели"
    ),
    schema_name: str = typer.Option(DEFAULT_SCHEMA_NAME, "--schema-name"),
    url: str = typer.Option(DEFAULT_URL, "--url", help="Адрес Qdrant"),
    collection: str = typer.Option(DEFAULT_COLLECTION, "--collection"),
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, "--embed-model"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8600, "--port"),
    no_serve: bool = typer.Option(
        False, "--no-serve", help="Остановиться после индексации, интерфейс не поднимать"
    ),
    model: list[str] = typer.Option(
        None, "--model", help="Переопределить модель (та же форма, что у convert)"
    ),
    env_file: Path = typer.Option(None, "--env-file", help="Путь к .env"),
    auth_db: Path = typer.Option(
        DEFAULT_AUTH_DB, "--auth-db", help="Файл учётных записей веб-интерфейса"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Подробный лог"),
) -> None:
    """Runs the whole pipeline: Excel -> SQLite -> passports -> Qdrant -> the interface."""
    _setup_logging(verbose)
    _load_env(env_file)
    if not no_serve:
        _require_accounts(auth_db)   # before the flow spends anything on the model

    if not os.environ.get("OPENAI_API_KEY"):
        typer.echo(
            "OPENAI_API_KEY не задан. Создайте .env со строкой OPENAI_API_KEY=sk-... "
            "или укажите файл через --env-file.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        router = build_router(model or ())
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    typer.echo("[1/2] Parser Agent: Excel -> SQLite -> паспорта -> Qdrant")
    try:
        state = _run_flow(
            {
                "input_path": str(input_path),
                "output_db": str(output_db),
                "output_dir": str(output_dir),
                "schema_name": schema_name,
                "describe_model": router.model_for(Task.DESCRIBE),
                "catalog_model": router.model_for(Task.CATALOG),
                "index": True,
                "qdrant_url": url,
                "collection": collection,
                "embed_model": embed_model,
            }
        )
    except (ConversionError, FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"      готово: {output_db}, строк {state.total_rows}, "
        f"точек {state.indexed_points} в {collection}"
    )
    _report_failures(state)   # the interface still opens over the sheets that did load

    if no_serve:
        typer.echo("[2/2] пропущено (--no-serve)")
        return

    typer.echo("[2/2] интерфейс")
    serve(
        db_path=output_db,
        catalog_dir=output_dir,
        host=host,
        port=port,
        url=url,
        collection=collection,
        embed_model=embed_model,
        model=model,
        env_file=env_file,
        auth_db=auth_db,
        verbose=verbose,
    )


@app.command()
def serve(
    db_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Файл SQLite"
    ),
    catalog_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, "--catalog-dir", help="Папка с <таблица>.catalog.json"
    ),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8600, "--port"),
    url: str = typer.Option(DEFAULT_URL, "--url", help="Адрес Qdrant для поиска"),
    collection: str = typer.Option(DEFAULT_COLLECTION, "--collection"),
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, "--embed-model"),
    model: list[str] = typer.Option(
        None, "--model", help="Переопределить модель (та же форма, что у convert)"
    ),
    env_file: Path = typer.Option(None, "--env-file", help="Путь к .env"),
    auth_db: Path = typer.Option(
        DEFAULT_AUTH_DB, "--auth-db", help="Файл учётных записей веб-интерфейса"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Подробный лог"),
) -> None:
    """Opens the web interface: the catalog of the database in a browser."""
    _setup_logging(verbose)
    _load_env(env_file)
    _require_accounts(auth_db)

    import uvicorn

    from .web import create_app

    try:
        router = build_router(model or ())
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    application = create_app(
        db_path,
        catalog_dir,
        qdrant_url=url,
        collection=collection,
        embed_model=embed_model,
        crew_models={
            "orchestrator": router.model_for(Task.ORCHESTRATOR),
            "sql": router.model_for(Task.SQL),
            "viz": router.model_for(Task.VIZ),
        },
        charts_dir=DEFAULT_OUTPUT_DIR / "charts",
        auth_db=auth_db,
    )
    from .crew.sandbox import image_ready

    if not image_ready():
        typer.echo(
            "Внимание: образ песочницы для графиков не собран — чат ответит, но без "
            "графиков. Собрать: python3 -m parser_agent sandbox build",
            err=True,
        )
    typer.echo(f"Интерфейс: http://{host}:{port}  (Ctrl+C — остановить)")
    uvicorn.run(application, host=host, port=port, log_level="warning")


@app.command()
def models(
    model: list[str] = typer.Option(
        None, "--model", help="Проверить, как повлияет переопределение (та же форма, что у convert)"
    ),
    env_file: Path = typer.Option(None, "--env-file", help="Путь к .env"),
) -> None:
    """Shows which model serves which stage and where the value came from."""
    _load_env(env_file)
    try:
        router = build_router(model or ())
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    width = max(len(task.value) for task in Task)
    typer.echo(f"{'ЗАДАЧА'.ljust(width)}  МОДЕЛЬ                ИСТОЧНИК")
    for task, model_name, source in router.table():
        typer.echo(f"{task.ljust(width)}  {model_name.ljust(20)}  {source}")


@app.command()
def prompt(
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True,
                                      help=INPUT_HELP),
    sheet: str = typer.Option(None, "--sheet", help="Только один лист (по умолчанию — все)"),
    skip_aggregation: bool = typer.Option(
        False, "--skip-aggregation", help="Показать промпт без модуля 3"
    ),
    model: list[str] = typer.Option(None, "--model", help="Та же форма, что у convert"),
    out: Path = typer.Option(
        None, "--out", help="Каталог для сохранения в файлы вместо печати в терминал"
    ),
    env_file: Path = typer.Option(None, "--env-file", help="Путь к .env"),
) -> None:
    """Prints the request that would go to the LLM at stage 2 — without calling the API.

    It shows exactly what is sent: the system prompt, the user message with the
    compressed sheet and the call parameters. No OpenAI key is needed.
    """
    _load_env(env_file)
    try:
        router = build_router(model or ())
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    model_name = router.model_for(Task.DESCRIBE)

    try:
        with prepared_workbook(input_path) as source:
            values_wb = source.load()
            available = source.sheet_names(values_wb)
    except ConversionError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    try:
        names = available if sheet is None else [sheet]
        if sheet is not None and sheet not in available:
            typer.echo(
                f"Лист {sheet!r} не найден. Доступные: {', '.join(available)}",
                err=True,
            )
            raise typer.Exit(code=2)

        for sheet_name in names:
            ws = values_wb[sheet_name]
            row_count = non_empty_row_count(ws)
            if row_count < MIN_NON_EMPTY_ROWS:
                typer.echo(
                    f"# лист {sheet_name!r} пропущен: непустых строк {row_count}", err=True
                )
                continue

            compressed, stats = compress_sheet_with_stats(
                ws, skip_aggregation=skip_aggregation
            )
            layout = detect_header_layout(ws)
            block = format_header_stacks(header_stacks(ws, layout)) if layout.detected else None
            user_message = build_user_prompt(compressed, sheet_name, block)
            total_tokens = estimate_tokens(SYSTEM_PROMPT + user_message)

            text = "\n".join(
                [
                    f"===== ЛИСТ: {sheet_name} =====",
                    f"model={model_name}  temperature=0  response_format=json_object",
                    f"сжатие: {stats.original_tokens} -> {stats.compressed_tokens} токенов "
                    f"(ratio {stats.ratio:.2f}x); весь запрос ~{total_tokens} токенов",
                    "",
                    "----- messages[0] role=system -----",
                    SYSTEM_PROMPT,
                    "",
                    "----- messages[1] role=user -----",
                    user_message,
                    "",
                ]
            )

            if out is None:
                typer.echo(text)
            else:
                out.mkdir(parents=True, exist_ok=True)
                path = out / f"{sanitize_table_name(sheet_name)}.prompt.txt"
                path.write_text(text, encoding="utf-8")
                typer.echo(f"[{sheet_name}] промпт сохранён: {path}")
    finally:
        values_wb.close()


users_app = typer.Typer(help="Учётные записи веб-интерфейса (создаёт администратор).")
app.add_typer(users_app, name="users")

AUTH_DB_OPTION = typer.Option(DEFAULT_AUTH_DB, "--auth-db", help="Файл учётных записей")


def _new_password() -> str:
    return typer.prompt("Пароль", hide_input=True, confirmation_prompt="Ещё раз")


TABLES_OPTION = typer.Option(
    None, "--tables",
    help="Только эти таблицы, через запятую (например sales,costs). Без флага — все таблицы",
)
DB_OPTION = typer.Option(
    Path("output.db"), "--db", help="База, по которой проверяются имена таблиц"
)


def _parse_tables(tables: str | None, db: Path) -> list[str] | None:
    """The --tables list, checked against the database: a typo must not leave
    the account with access to nothing — or to a table that appears later."""
    if tables is None:
        return None
    names = sorted({name.strip() for name in tables.split(",") if name.strip()})
    if db.exists():
        connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            existing = set(list_tables(connection))
        finally:
            connection.close()
        unknown = [name for name in names if name not in existing]
        if unknown:
            typer.echo(
                f"Таких таблиц в {db} нет: {', '.join(unknown)}. Есть: {', '.join(sorted(existing))}",
                err=True,
            )
            raise typer.Exit(code=2)
    else:
        typer.echo(f"Внимание: {db} не найдена, имена таблиц не проверены", err=True)
    return names


def _describe_access(tables: list[str] | None) -> str:
    if tables is None:
        return "все таблицы"
    return ", ".join(tables) if tables else "ни одной таблицы"


@users_app.command("add")
def users_add(
    username: str,
    tables: str = TABLES_OPTION,
    db: Path = DB_OPTION,
    auth_db: Path = AUTH_DB_OPTION,
) -> None:
    """Creates an account; the password is asked for twice and never echoed."""
    allowed = _parse_tables(tables, db)
    try:
        check_username(username)
        store = AuthStore(auth_db)
        store.add_user(username, _new_password())
        store.set_access(username, allowed)
    except AuthError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"Пользователь {username} создан ({auth_db}); доступ: {_describe_access(allowed)}")


@users_app.command("access")
def users_access(
    username: str,
    tables: str = TABLES_OPTION,
    all_tables: bool = typer.Option(False, "--all", help="Открыть все таблицы"),
    db: Path = DB_OPTION,
    auth_db: Path = AUTH_DB_OPTION,
) -> None:
    """Sets which tables an account may read: --tables a,b or --all."""
    if all_tables == (tables is not None):
        typer.echo("Укажите либо --tables a,b, либо --all", err=True)
        raise typer.Exit(code=2)
    allowed = None if all_tables else _parse_tables(tables, db)
    try:
        AuthStore(auth_db).set_access(username, allowed)
    except AuthError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"Доступ {username}: {_describe_access(allowed)} (действует со следующего вопроса)")


@users_app.command("passwd")
def users_passwd(username: str, auth_db: Path = AUTH_DB_OPTION) -> None:
    """Sets a new password; the user is signed out everywhere."""
    try:
        AuthStore(auth_db).set_password(username, _new_password())
    except AuthError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"Пароль {username} изменён, все его сессии закрыты")


@users_app.command("remove")
def users_remove(username: str, auth_db: Path = AUTH_DB_OPTION) -> None:
    """Deletes an account together with its sessions."""
    if not AuthStore(auth_db).remove_user(username):
        typer.echo(f"Пользователя {username} нет", err=True)
        raise typer.Exit(code=2)
    typer.echo(f"Пользователь {username} удалён")


@users_app.command("list")
def users_list(auth_db: Path = AUTH_DB_OPTION) -> None:
    """Lists the accounts."""
    rows = AuthStore(auth_db).list_users()
    if not rows:
        typer.echo("Пользователей нет. Создать: python3 -m parser_agent users add <имя>")
    for username, created_at, tables in rows:
        typer.echo(f"{username}\tсоздан {created_at}\tдоступ: {_describe_access(tables)}")


sandbox_app = typer.Typer(help="Docker-песочница, в которой агент рисует графики.")
app.add_typer(sandbox_app, name="sandbox")


@sandbox_app.command("build")
def sandbox_build() -> None:
    """Builds the chart sandbox image (python + pandas + matplotlib), once."""
    from .crew.sandbox import IMAGE, build_image

    if build_image().returncode != 0:
        typer.echo(f"Не удалось собрать {IMAGE}", err=True)
        raise typer.Exit(code=2)
    typer.echo(f"Образ {IMAGE} собран")


@sandbox_app.command("check")
def sandbox_check() -> None:
    """Draws a test chart in the sandbox: shows that Docker and the image work."""
    import tempfile

    from .crew.sandbox import run_chart_code

    with tempfile.TemporaryDirectory() as folder:
        result = run_chart_code(
            "plt.bar(df['item'], df['value']); plt.title('sandbox check'); plt.savefig(OUT)",
            ["item", "value"], [["a", 1], ["b", 3], ["c", 2]], Path(folder),
        )
    if not result.ok:
        typer.echo(f"Песочница не работает: {result.error}", err=True)
        raise typer.Exit(code=2)
    typer.echo("Песочница работает: тестовый график нарисован")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
