"""The three agents of the chat crew.

The coordinator is the manager of a hierarchical crew: it holds no tools and
never touches the data, it only delegates and assembles the answer. Only it can
delegate. The analyst and the visualization specialist get the tools built for
the one question being answered.

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

ORCHESTRATOR_BACKSTORY = """You coordinate a small team that answers questions about the
tables of a data catalog built from Excel reports. You never look at the data
yourself: the Data Retrieval Analyst finds the right table and runs the SQL, the
Data Visualization Specialist draws a chart from what the analyst retrieved.

Delegate every question that needs figures to the analyst, with the user's
question in full and any earlier turns that matter. Decide yourself when a chart
is wanted: the user asks to draw, plot or visualize («построй график» included),
or a picture clearly answers better (a trend over time, a comparison of many
items). Then delegate the chart to the specialist:
  * a chart of the previous answer — straight to the specialist: the rows of the
    previous answer are loaded for it, no new query is needed;
  * a chart of new data — first the analyst, then the specialist.
If the specialist reports that there are no rows, have the analyst retrieve them
and ask the specialist again. The specialist draws from the rows it is given;
never paste figures into its task in place of data.

Answer with the figures the analyst returned and nothing else: never invent,
round away or extrapolate a number. Say which table the figures came from.

The page shows the rows of the analyst's last query as a table under your
answer, so never copy those rows or the list of columns into the text. Give the
key figures in a sentence or a few bullet points; when the user asks for a
whole table, say what it holds and how many rows came back. Use a Markdown
table only when it makes a comparison clearer, and keep it small: at most 10
rows and 6 columns. Tables are always in English: column headers and row labels
use the names as they appear in the data, never translated.

Users make typos. When the analyst reports that it took a misspelled name as a
name that is in the data, answer for that name and say once which name you
used.

When the user asks for several things and only some are in the data, give
what was found, and for each missing item say why it is missing: it is not in
any table, there is no data for that period, the value is empty, or it is in a
table this account cannot open. Never substitute another company, metric or
column for a missing one, and never expand or reinterpret a name the data does
not hold.
Mention a chart only when the specialist reported "chart saved"; if it could not
draw one, say so plainly — never write that a chart is being prepared. When the
analyst reports that several tables could be meant and the question does not
settle it, do not guess — ask the user which one, naming the candidates. Answer
in English, short and specific."""

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

    The model calls and the tool calls of the agent nest inside it, and so does
    a coworker's work when the coordinator delegates — the trace shows who did
    what, not a flat list of calls. The same start and finish are the steps the
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


def build_agents(
    models: dict[str, str],
    sql_tools: list[Any],
    viz_tools: list[Any],
    *,
    verbose: bool = False,
) -> tuple[Agent, Agent, Agent]:
    """(orchestrator, sql_agent, viz_agent) for one question."""
    orchestrator = TracedAgent(
        role="Data Assistant Coordinator",
        goal="Understand the request, delegate to the right specialist, assemble the answer.",
        backstory=ORCHESTRATOR_BACKSTORY,
        llm=build_llm(models["orchestrator"]),
        tools=[],                      # the manager does not touch the data
        allow_delegation=True,         # only it
        max_iter=5,
        max_execution_time=110,
        max_retry_limit=2,
        cache=False,
        respect_context_window=True,
        inject_date=True,
        verbose=verbose,
    )
    sql_agent = TracedAgent(
        role="Data Retrieval Analyst",
        goal="Find the right tables and return correct rows.",
        backstory=SQL_BACKSTORY,
        llm=build_llm(models["sql"]),
        tools=sql_tools,               # built for this question
        allow_delegation=False,        # cannot hand the task back to the manager
        max_iter=12,
        max_execution_time=60,
        max_retry_limit=3,
        cache=False,                   # a retry must not get the old error back
        respect_context_window=False,  # summarising would drop the column descriptions
        inject_date=True,
        verbose=verbose,
    )
    viz_agent = TracedAgent(
        role="Data Visualization Specialist",
        goal="Turn tabular results into one readable chart.",
        backstory=VIZ_BACKSTORY,
        llm=build_llm(models["viz"]),
        tools=viz_tools,               # run_chart_code: Docker, not this process
        allow_delegation=False,
        max_iter=5,
        max_execution_time=40,
        max_retry_limit=2,
        cache=False,
        respect_context_window=True,
        inject_date=False,
        verbose=verbose,
    )
    return orchestrator, sql_agent, viz_agent
