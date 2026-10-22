"""One chat question as the Flow carries it, and what the agents share about answers.

ChatRun is the question in the Flow's state (flow.py): web.py fills it, the Jev
tree and the Jev loop read it, the loop writes the answer into it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# How many earlier turns of the conversation the agents are shown.
MAX_HISTORY_TURNS = 6
MAX_TURN_CHARS = 2000


class ChatRun(BaseModel):
    """One question for the chat process of the flow, and its answer."""

    question: str
    history: list[dict[str, str]] = Field(default_factory=list)
    previous_sql: str | None = None          # the query behind the previous answer
    previous_table: str | None = None
    db_path: str
    catalog_dir: str
    charts_dir: str
    qdrant_url: str
    collection: str
    embed_model: str
    models: dict[str, str]                   # orchestrator (the answer analyst) / sql / viz
    allowed_tables: list[str] | None = None  # None — the user reads every table
    answer: dict[str, Any] | None = None


EXPECTED_OUTPUT = (
    "A short, specific answer for the user in English: the figures, what they are "
    "and which table they came from. Do not copy the query rows into the text: the "
    "page shows them as a table under the answer. A Markdown table only for a small "
    "comparison (at most 10 rows and 6 columns), in English, with the names as they "
    "appear in the data. If a name was misspelled, say which name "
    "from the data you used. If only part of what was asked is in the data, give "
    "that part and say for each missing item why it is missing — never substitute "
    "something else for it. Mention a chart only if the Data Visualization "
    "Specialist reported \"chart saved\"; if it could not draw one, say so. If the "
    "question cannot be answered from the data, say why; if it is unclear which "
    "table is meant, ask one clarifying question."
)
