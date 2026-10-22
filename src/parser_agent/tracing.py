"""Langfuse tracing — on when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.

What one workbook sends to Langfuse (LANGFUSE_BASE_URL — the stack in langfuse/):

    trace "parser_agent: <workbook>"        session = one launch of the CLI
    ├── prepare, compress, load, index      the flow steps, as spans
    ├── describe
    │   └── sheet <name>
    │       └── OpenAI-generation           the prompt, the answer, tokens, cost
    └── passports
        └── sheet <name> ...
    scores: sheets_loaded, sheets_failed

The generations come from `langfuse.openai`: importing it patches the openai
package itself, so describer, catalog, chat and indexer are traced unchanged.

Langfuse gets a TracerProvider of its own. OpenTelemetry keeps one global
provider, and CrewAI registers its telemetry exporter (telemetry.crewai.com)
in it; had Langfuse joined that provider, every prompt — the compressed sheets
— would have been exported to CrewAI's telemetry as well.

Without the keys nothing here imports Langfuse: the decorators and context
managers do nothing, and the project works exactly as without tracing.
"""

from __future__ import annotations

import functools
import inspect
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_client: Any = None

# One launch of the CLI is one session: `crewai run` over several workbooks
# groups their traces together.
_SESSION_ID = f"run-{datetime.now():%Y-%m-%d_%H-%M-%S}"


def enabled() -> bool:
    """Whether the keys are set. Meaningful only after .env has been loaded."""
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def init() -> Any:
    """Starts Langfuse once. Call it after .env is loaded and before the first
    model call. Returns the client, or None when tracing is off."""
    global _client
    if _client is None and enabled():
        from langfuse import Langfuse
        from opentelemetry.sdk.trace import TracerProvider

        _client = Langfuse(tracer_provider=TracerProvider())
        import langfuse.openai  # noqa: F401 — patches openai: every call becomes a generation
    return _client


def observe(name: str) -> Callable[[F], F]:
    """A span around a function, when tracing is on.

    Arguments and results are not captured: for a flow step they are the flow
    itself, with the whole workbook in memory. What a step did is written with
    `update(output=...)` instead.
    """

    def decorate(func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if _client is None:
                    return await func(*args, **kwargs)
                with _client.start_as_current_observation(name=name):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _client is None:
                return func(*args, **kwargs)
            with _client.start_as_current_observation(name=name):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


@contextmanager
def span(name: str, **fields: Any) -> Iterator[Any]:
    """A span around a part of a function — one sheet inside a step, say."""
    if _client is None:
        yield None
        return
    with _client.start_as_current_observation(name=name, **fields) as observation:
        yield observation


def update(**fields: Any) -> None:
    """Writes output, metadata or an error level into the current span."""
    if _client is not None:
        _client.update_current_span(**fields)


@contextmanager
def trace(
    name: str,
    *,
    title: str,
    input: Any = None,
    tags: list[str] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> Iterator[Any]:
    """The root span of a trace: everything started inside nests under it.

    title becomes the trace name in the Langfuse list. session_id groups traces:
    one chat conversation, or — by default — one launch of the CLI. user_id is
    whoever is signed in to the web interface. Both must be US-ASCII.
    """
    if _client is None:
        yield None
        return
    from langfuse import propagate_attributes

    with _client.start_as_current_observation(name=name, input=input) as root:
        with propagate_attributes(
            trace_name=title, session_id=session_id or _SESSION_ID,
            user_id=user_id, tags=tags or [],
        ):
            yield root


def workbook_trace(inputs: dict[str, Any]) -> Any:
    """The trace of one workbook through the flow."""
    workbook = Path(str(inputs.get("input_path", ""))).name
    return trace(
        "parser_agent",
        title=f"parser_agent: {workbook}",
        input={key: value for key, value in inputs.items() if value is not None},
        tags=["dry-run"] if inputs.get("dry_run") else ["parser_agent"],
    )


def finish_workbook(root: Any, state: Any) -> None:
    """The outcome of a workbook: the trace output and two scores."""
    if root is None:
        return
    failed = {sheet.sheet_name: sheet.error for sheet in state.failed_sheets}
    root.update(
        output={
            "database": state.output_db,
            "rows": state.total_rows,
            "sheets_loaded": [sheet.sheet_name for sheet in state.loaded_sheets],
            "sheets_failed": failed,
        },
        level="WARNING" if failed else None,
    )
    if not state.dry_run:
        root.score_trace(name="sheets_loaded", value=len(state.loaded_sheets))
        root.score_trace(name="sheets_failed", value=len(failed))


def flush() -> None:
    """Sends what is buffered: a CLI run ends right after the flow."""
    if _client is not None:
        _client.flush()
