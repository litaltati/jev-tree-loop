"""The three agents of the loop, each run on its own with its task built from the state.

The SQL agent and the visualizer are built from the instructions and tools in
crew/; each gets a task of its own, written from the LoopState, so every agent starts with the user's question and the conversation,
and the visualizer and the answer analyst with what the agents before them found.
"""

from __future__ import annotations

from typing import Any

from crewai import Crew, Process, Task

from ..chat import rows_for_prompt
from ..crew.agents import SQL_BACKSTORY, VIZ_BACKSTORY, TracedAgent, build_llm
from ..crew.chat_crew import EXPECTED_OUTPUT, MAX_HISTORY_TURNS, MAX_TURN_CHARS
from .state import LoopState

ANALYST_ROLE = "Answer Analyst"

ANALYST_BACKSTORY = """You write the final answer of a small team that answers questions
about the tables of a data catalog built from Excel reports. You never query the
data yourself: the Data Retrieval Analyst found the table and ran the SQL, and
the Data Visualization Specialist drew a chart when one was wanted. You get the
user's question, the conversation, the analyst's report, the rows its query
returned and whether a chart was saved.

Answer with the figures in those rows and nothing else: never invent, round away
or extrapolate a number. Say which table the figures came from. The page shows
the rows as a table under your answer, so never copy them or the list of
columns into the text: give the key figures in a sentence or a few bullet
points. Use a Markdown table only when it makes a comparison clearer, at most 10
rows and 6 columns, in English, with the names as they appear in the data.

When the analyst took a misspelled name for a name in the data, answer for that
name and say once which name was used. When only part of what was asked was
found, give that part and say for each missing item why it is missing — never
substitute another company, metric or column for it. When the analyst reports
that several tables could be meant, ask the user which one, naming them. When
the question needs no data — a definition, small talk, the previous answer put
another way — answer it briefly yourself, without figures you were not given.
Mention a chart only when it was saved. Answer in English, short and specific."""


def _conversation(state: LoopState) -> list[str]:
    turns = [t for t in state.run.history if t.get("content")][-MAX_HISTORY_TURNS:]
    lines = []
    if turns:
        lines.append("\nEarlier in this conversation:")
        for turn in turns:
            who = "Assistant" if turn.get("role") == "assistant" else "User"
            lines.append(f"- {who}: {str(turn['content'])[:MAX_TURN_CHARS]}")
    return lines


def _previous_query(state: LoopState) -> list[str]:
    run = state.run
    if not run.previous_sql:
        return []
    return [f"\nThe previous answer came from this query on the table {run.previous_table or '(unknown)'}:\n"
            f"{run.previous_sql}"]


async def _run_one(state: LoopState, agent: TracedAgent, description: str, expected: str) -> str:
    task = Task(description=description, expected_output=expected, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    output = await crew.kickoff_async()          # no inputs: a brace in the question stays as it is
    state.agent_tokens += int(getattr(output.token_usage, "total_tokens", 0) or 0)
    return (output.raw or "").strip()


async def run_sql_agent(state: LoopState, tools: list[Any]) -> str:
    lines = [f'A user of the data catalog asks: "{state.question}"', *_conversation(state), *_previous_query(state)]
    if state.runs_of("sql") > 1:
        lines.append(f"\nA first attempt did not answer the question. Its report:\n{state.reports.get('sql', '')}\n"
                     "Look again: another table, the exact spelling from the passport, other filters.")
    lines.append("\nFind the table with catalog_search, then answer with ONE SQLite query run through sql_query.")
    agent = TracedAgent(
        role="Data Retrieval Analyst", goal="Find the right tables and return correct rows.",
        backstory=SQL_BACKSTORY, llm=build_llm(state.run.models["sql"]), tools=tools,
        allow_delegation=False, max_iter=12, max_execution_time=60, max_retry_limit=3,
        cache=False, respect_context_window=False, inject_date=True,
    )
    return await _run_one(state, agent, "\n".join(lines),
                          "The table name, the SQL you ran, what its rows hold, and which names were "
                          "misspelled or are missing.")


async def run_viz_agent(state: LoopState, tools: list[Any]) -> str:
    query = state.ctx.last_query
    if query is not None:
        rows = f"`df` holds the rows of this query on the table {query.get('table') or '?'}: {query['sql']}\n" \
               f"columns: {query['columns']} · {len(query['rows'])} rows"
    else:
        rows = "`df` holds the rows of the previous answer of the conversation."
    description = "\n".join([f'A user of the data catalog asks: "{state.question}"', *_conversation(state),
                             f"\n{rows}", "\nDraw the one chart that answers the question and save it."])
    agent = TracedAgent(
        role="Data Visualization Specialist", goal="Turn tabular results into one readable chart.",
        backstory=VIZ_BACKSTORY, llm=build_llm(state.run.models["viz"]), tools=tools,
        allow_delegation=False, max_iter=5, max_execution_time=40, max_retry_limit=2,
        cache=False, respect_context_window=True, inject_date=False,
    )
    return await _run_one(state, agent, description, '"chart saved", or what went wrong.')


async def run_analyst(state: LoopState) -> str:
    query = state.ctx.last_query
    found = []
    # The tree's notes, unless the SQL agent has since replaced the rows they are about.
    if "tree" in state.reports and not (state.tree_found and state.runs_of("sql")):
        found.append(f"A direct lookup in the report tables noted:\n{state.reports['tree']}")
    if "sql" in state.reports:
        found.append(f"The Data Retrieval Analyst reported:\n{state.reports['sql']}")
    if query is not None:
        found.append(f"The query, on the table {query.get('table') or '?'}:\n{query['sql']}\n"
                     f"The rows it returned (the page shows them under your answer):\n{rows_for_prompt(query)}")
    elif "sql" in state.reports:
        found.append("Its query returned no rows.")
    if state.chart_tried:
        found.append("A chart was saved." if state.ctx.chart_id else
                     f"No chart could be drawn: {state.reports.get('visualizer', '')}")
    if not found:
        found.append("Nothing was looked up: the question needs no data.")
    description = "\n".join([f'A user of the data catalog asks: "{state.question}"', *_conversation(state),
                             "\nWhat the team found:", *found, "\nWrite the answer for the user."])
    agent = TracedAgent(
        role=ANALYST_ROLE, goal="Answer the user from what the team found.",
        backstory=ANALYST_BACKSTORY, llm=build_llm(state.run.models["orchestrator"]), tools=[],
        allow_delegation=False, max_iter=3, max_execution_time=60, max_retry_limit=2,
        cache=False, respect_context_window=True, inject_date=True,
    )
    return await _run_one(state, agent, description, EXPECTED_OUTPUT)
