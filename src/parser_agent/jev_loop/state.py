"""The one state every step of the loop reads and adds to."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..crew.chat_crew import ChatRun
from ..crew.tools import ChatContext


@dataclass
class LoopState:
    """One question on its way through the loop.

    run      the question, the conversation, where the data is (from web.py)
    ctx      what the tools found: the rows of the last query and the chart —
             the same object the SQL agent's and the visualizer's tools write to
    reports  what each agent said when it finished, by step ("sql", "visualizer")
    steps    the steps taken, in order, with Jev's probabilities
    tree_route  the route of the Jev tree, when it was sure of it (jev_tree/tree.py)
    first_step  the step the tree already knows comes first ("sql", "exit"); None — Jev decides
    tree_found  the rows in ctx.last_query are the tree's: it found the figures itself
    """

    run: ChatRun
    ctx: ChatContext
    reports: dict[str, str] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    jev_calls: list[dict[str, Any]] = field(default_factory=list)
    agent_tokens: int = 0
    tree_route: str | None = None
    first_step: str | None = None
    tree_found: bool = False

    @property
    def question(self) -> str:
        return self.run.question

    def runs_of(self, step: str) -> int:
        return sum(1 for s in self.steps if s["step"] == step)

    @property
    def rows(self) -> list[list[Any]]:
        return (self.ctx.last_query or {}).get("rows") or []

    @property
    def chart_tried(self) -> bool:
        return self.runs_of("visualizer") > 0
