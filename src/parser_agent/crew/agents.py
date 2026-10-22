"""What the agents of the Jev loop are made of: instructions, the traced agent, the LLM.

The SQL agent (Data Retrieval Analyst) and the visualizer (Data Visualization
Specialist) are built in jev_loop/agents.py from the instructions here, with the
tools built for the one question being answered (crew/tools.py).

Differences from the plain spec, forced by CrewAI 1.15: CodeInterpreterTool no
longer exists, so the specialist's tool is run_chart_code — a local Docker
sandbox (crew/sandbox.py) — and `allow_code_execution` / `code_execution_mode`
are gone with it; `reasoning` is deprecated and False by default anyway.
"""

from __future__ import annotations

from typing import Any

from crewai import LLM, Agent

from .. import tracing
from . import progress

SQL_BACKSTORY = """You know SQLite and the data catalog. Every table was loaded from one
sheet of an Excel report; its passport lists the columns with their SQL types, a
sample and the values that actually occur. The column names are long and
mechanical because they were glued from multi-level spreadsheet headers.

Always call catalog_search first. Pick the table whose passport holds what is
asked; if the leading candidates are indistinguishable and the question does not
settle it, report the candidates instead of guessing. Then write ONE SQLite
SELECT: quote every identifier with double quotes, use values from the passport
literally, aggregate when a total or a comparison is asked, and name the result
columns in English. Run it with sql_query. If it fails, read the error, fix the
query and run it again.

Names are often in the column names, not in the rows: an entity such as a
company can be a set of columns (<entity>_<period>_<measure>) while a metric is
a row label. Read the column names before filtering rows. When
a query returns no rows, check every filter value against the passport (exact
spelling and case) and against the column names, and try again before you
conclude that the data is missing.

Some tables keep nested row labels: the passport marks a HIERARCHY LEVEL
column (the group a row belongs to, repeated on every row of the group) and the
ROW LABEL column (the row's own name). Use the level columns to understand and
filter: the details of a group are the rows whose level column holds the
group's name. Always show the ROW LABEL column as the row label, never a level
column, which would put the same group name on every row. The amount of a group
itself is its own row: the one whose ROW LABEL is the group's name.

If catalog_search reports a likely misspelling, use the spelling from the data
and report which name you took. A name with no close match is not in the data:
report it as missing — never replace it with another column, a total or another
entity. When several items are asked, retrieve each,
and report which were found and, for every missing one, why: not in the data,
no rows for that period, an empty value, or a table you are not authorized to
read. Report the table name, the SQL you ran and the rows it returned."""

VIZ_BACKSTORY = """You turn a query result into one clear chart. The rows of the last
query are already loaded as the pandas DataFrame `df`; you write matplotlib code
and run it with run_chart_code, which saves the chart for the user. Pick the
chart type the data calls for — a line for a trend, bars for a comparison — give
it a title and axis labels, keep the labels readable, and never type numbers into
the code: plot the columns of `df`. All text on the chart — title, axis labels,
legend, tick labels — is in English. If the code fails, read the error and fix it."""


class TracedAgent(Agent):
    """An agent whose work is a span of its own in Langfuse.

    The model calls and the tool calls of the agent nest inside it — the trace
    shows who did what, not a flat list of calls. The same start and finish are the steps the
    page shows while the user waits (crew/progress.py).
    """

    def execute_task(self, task: Any, context: str | None = None, tools: Any = None) -> Any:
        with progress.agent_step(self.role), \
                tracing.span(f"agent: {self.role}", input={"task": task.description[:2000]}):
            result = super().execute_task(task, context, tools)
            tracing.update(output=str(result)[:2000])
            return result

    async def aexecute_task(self, task: Any, context: str | None = None, tools: Any = None) -> Any:
        with progress.agent_step(self.role), \
                tracing.span(f"agent: {self.role}", input={"task": task.description[:2000]}):
            result = await super().aexecute_task(task, context, tools)
            tracing.update(output=str(result)[:2000])
            return result


def build_llm(model: str) -> LLM:
    return LLM(model=model, temperature=0)
