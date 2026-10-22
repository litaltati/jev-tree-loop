"""Jev decides the next step; code decides which steps it may choose from.

The Jev tree goes first (jev_tree/); then one Choice per turn of the loop, with
the state as it is by then: sql, visualizer or exit — exit hands the state to
the answer analyst. Before asking, code strikes out what cannot be right:

  * visualizer — only when there are rows to draw: of a query of this loop, or
    of the previous answer of the conversation; and only once;
  * sql        — at most twice (a second try after an empty or failed one), and
                 not once the tree has found the rows: its lookup is checked by
                 code (names by spelling, the line, columns from the passport);
  * after the chart there is nothing left but exit.

When a single step is left, Jev is not asked at all; nor for the first step
when the tree already knows it: the SQL agent for more than one line, a ranking,
a calculation or a table name, the answer for a definition, small talk or a
company the data lacks.
Without Jev (no key, or it cannot be reached) code takes the steps: the SQL
agent once, then the answer.
"""

from __future__ import annotations

import re
import time
from typing import Any

from typesafe_sdk import Choice, TypeSafeClient

from .. import tracing
from .state import LoopState

USD_PER_INPUT_TOKEN = 0.042 / 1_000_000
MAX_SQL_RUNS = 2
MAX_STEPS = 4            # sql, sql, visualizer, exit — the longest path there is


STEPS = {
    "sql": "Look up figures in the database: the question needs data, and nothing retrieved "
           "so far answers it (nothing yet, an empty result, or the wrong figures).",
    "visualizer": "Draw a chart of the rows already retrieved: the user asks for a chart, graph, "
                  "plot or picture — in any language — or to draw the previous answer.",
    "exit": "Stop and write the answer now: the rows retrieved answer the question, or the "
            "question needs no figures at all (a definition, small talk, the previous answer "
            "put another way), or the chart has been drawn.",
}


def _masked(text: Any, limit: int = 240) -> str:
    # Figures stay on the machine: every number of two digits or more becomes "#".
    return re.sub(r"\d{2,}", "#", " ".join(str(text).split()))[:limit]


def allowed_steps(state: LoopState) -> list[str]:
    if state.chart_tried:
        return ["exit"]
    steps = ["exit"]
    tree_rows = state.tree_found and bool(state.rows)
    if not tree_rows and state.runs_of("sql") < MAX_SQL_RUNS:
        steps.append("sql")
    has_rows = bool(state.rows) or (state.ctx.last_query is None and state.ctx.previous_query is not None)
    if has_rows:
        steps.append("visualizer")
    return steps


def steps_taken(state: LoopState) -> list[str]:
    return (["tree"] if state.tree_found else []) + [s["step"] for s in state.steps]


def describe(state: LoopState) -> dict[str, Any]:
    """What Jev is told about the work so far — a description, never the figures."""
    query = state.ctx.last_query
    if query is None:
        data = "nothing retrieved yet"
        if state.ctx.previous_query is not None:
            data += " in this turn; the previous answer of the conversation has rows that can be drawn"
    else:
        labels = [str(row[0]) for row in query["rows"][:10] if row]
        data = (f"retrieved already: table {query.get('table') or '?'} · {len(query['rows'])} rows · "
                f"columns: {', '.join(map(str, query['columns']))}"
                + (f" · first column: {', '.join(labels)}" if labels else ""))
    if state.runs_of("sql") and not state.rows:
        data += " · the last SQL run returned no rows"
    return {
        "message": state.question,
        "conversation": [
            {"role": t.get("role", "user"),
             "content": t["content"] if t.get("role") == "user" else _masked(t["content"])}
            for t in state.run.history if t.get("content")
        ][-6:],
        "progress": {"steps_taken": steps_taken(state), "data": _masked(data, 600),
                     "chart": "not drawn",
                     **({"lookup": _masked(state.reports["tree"], 400)} if "tree" in state.reports else {})},
    }


def next_step(client: TypeSafeClient | None, state: LoopState) -> str:
    """The step to take now. Jev chooses among the steps code still allows."""
    allowed = allowed_steps(state)
    turn = len(state.steps) + 1
    if len(allowed) == 1:
        state.steps.append({"step": allowed[0], "by": "code", "allowed": allowed})
        return allowed[0]
    by_tree = state.first_step if not state.steps else None
    if by_tree in allowed:
        state.steps.append({"step": by_tree, "by": "tree", "route": state.tree_route, "allowed": allowed})
        return by_tree
    if client is None:
        step = "sql" if not state.steps and "sql" in allowed else "exit"
        state.steps.append({"step": step, "by": "code", "allowed": allowed, "note": "no Jev"})
        return step

    jev_state = describe(state)
    question = Choice(
        instructions="Read `message` together with `conversation` and `progress`. A data assistant "
                     "over the company's financial reports is answering the user. Which step should "
                     "it take next? A short reply — a table name, a correction — continues the "
                     "user's earlier request.",
        criteria={step: STEPS[step] for step in allowed},
    )
    with tracing.span(f"jev: next step #{turn}", as_type="generation", model="jev-latest",
                      input={"state": jev_state, "allowed": allowed}) as call:
        started = time.perf_counter()
        response = client.system_one(jev_state, {"next": question})
        ms = (time.perf_counter() - started) * 1000
        tokens = int(getattr(getattr(response, "usage", None), "input_tokens", 0) or 0)
        if call is not None:
            call.update(model=response.model, output=response.model_dump(mode="json"),
                        usage_details={"input": tokens}, cost_details={"input": tokens * USD_PER_INPUT_TOKEN},
                        metadata={"latency_ms": round(ms)})
    answer = response.answers["next"]
    choice = answer.choice if answer.choice in allowed else "exit"
    state.jev_calls.append({"turn": turn, "ms": round(ms), "tokens": tokens})
    state.steps.append({"step": choice, "by": "jev", "allowed": allowed,
                        "confidence": round(answer.confidence, 3),
                        "probabilities": {k: round(v, 3) for k, v in answer.probabilities.items()}})
    return choice
