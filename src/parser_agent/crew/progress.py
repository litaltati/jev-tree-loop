"""What the crew is doing right now — the steps the page shows while it waits.

    ✓ Understanding the question     the coordinator reads it, until it delegates
    ✓ Finding the data               the Data Retrieval Analyst at work
    ◌ Building the chart             the Data Visualization Specialist at work
      Preparing the answer           the coordinator again, after its coworkers

The steps come from the agents themselves (TracedAgent) and from the chart tool,
not from guesses: a step is ticked when its agent has finished, and the chart is
ticked only when the sandbox has actually saved one.

The listener of the question being answered sits in a context variable, like the
Langfuse span: CrewAI copies the context into the threads it runs the agents
in, so two users asking at once each get the steps of their own question.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator

logger = logging.getLogger(__name__)

QUESTION, DATA, CHART, ANSWER = "question", "data", "chart", "answer"
RUNNING, DONE, FAILED = "running", "done", "failed"

# The coworker steps, by agent role (crew/agents.py). The coordinator's own work
# is QUESTION before it delegates and ANSWER after.
STEP_BY_ROLE = {
    "Data Retrieval Analyst": DATA,
    "Data Visualization Specialist": CHART,
    "Answer Analyst": ANSWER,          # the Jev loop's last agent (jev_loop/agents.py)
}

Step = dict[str, str]                       # {"step": ..., "state": ...}
Listener = Callable[[list[Step]], None]     # gets every step so far, in order


class Progress:
    """The steps of one answer. Every change is sent to the listener whole."""

    def __init__(self, listener: Listener) -> None:
        self._listener = listener
        self._steps: dict[str, str] = {}
        self._chart_saved = False
        self._lock = threading.Lock()

    def begin(self) -> None:
        with self._lock:
            self._steps = {QUESTION: RUNNING}
            self._send()

    def agent_started(self, step: str) -> None:
        with self._lock:
            if self._steps.get(QUESTION) == RUNNING:
                self._steps[QUESTION] = DONE
            # "Preparing the answer" was premature: the coordinator delegates more.
            if self._steps.get(ANSWER) == RUNNING:
                del self._steps[ANSWER]
            if step == CHART:
                self._chart_saved = False
            self._steps.pop(step, None)       # a repeated step moves to the end
            self._steps[step] = RUNNING
            self._send()

    def agent_finished(self, step: str, ok: bool) -> None:
        with self._lock:
            if step == CHART:
                ok = ok and self._chart_saved
            self._steps[step] = DONE if ok else FAILED
            if step != ANSWER and RUNNING not in self._steps.values():
                self._steps[ANSWER] = RUNNING
            self._send()

    def chart_saved(self) -> None:
        with self._lock:
            self._chart_saved = True

    def _send(self) -> None:
        snapshot = [{"step": step, "state": state} for step, state in self._steps.items()]
        try:
            self._listener(snapshot)
        except Exception as exc:   # the page missing a step must not fail the answer
            logger.debug("progress listener failed: %s", exc)


_current: ContextVar[Progress | None] = ContextVar("parser_agent_progress", default=None)


@contextmanager
def reporting(progress: Progress) -> Iterator[Progress]:
    """The crew run inside reports its steps to `progress`."""
    token = _current.set(progress)
    try:
        progress.begin()
        yield progress
    finally:
        _current.reset(token)


@contextmanager
def agent_step(role: str) -> Iterator[None]:
    """Around one agent's task: its step runs, then is ticked or marked failed."""
    with code_step(STEP_BY_ROLE.get(role)):
        yield


@contextmanager
def code_step(step: str | None) -> Iterator[None]:
    """Around a step done by code instead of an agent — the Jev tree's lookup or
    its chart: the same step on the page, ticked the same way."""
    progress = _current.get()
    if progress is None or step is None:
        yield
        return
    progress.agent_started(step)
    ok = False
    try:
        yield
        ok = True
    finally:
        progress.agent_finished(step, ok)


def chart_saved() -> None:
    progress = _current.get()
    if progress is not None:
        progress.chart_saved()
