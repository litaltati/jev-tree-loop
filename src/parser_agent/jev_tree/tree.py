"""The decision tree: Jev answers bounded questions, the code builds the query.

    stage 1 — one call:  route · table · period · actual/budget · "all of them" · breakdown · ranking
                         · more than one line — the tree takes one line; more goes to the SQL agent
    stage 2 — one call per likely table (the two best):  the line · each entity (yes/no)
                                                         · an entity the table does not have
    then    — code:  the paths in order of likelihood, each table's line, its
              companies checked by spelling (names.py) and its columns; the
              first path that holds together gives the SQL

Every choice is one of the options read from the passports (options.py), plus
"none of these". Nothing writes SQL freely, so nothing can put another company
in place of one the data does not have: that is answered as "not in the data".
A route other than data / data + chart, an unsure choice or an empty path is
not answered here — it goes on to the Jev loop, or to the user as a question.
"""

from __future__ import annotations

import contextvars
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

from typesafe_sdk import Choice, Noul, TypeSafeClient

from .. import tracing
from ..catalog import quote_identifier
from .names import CLOSE, name_key, reconcile, spelling, user_words
from .options import MONTHS, TableOptions

USD_PER_INPUT_TOKEN = 0.042 / 1_000_000

ROUTE_GATE = 0.5      # below it the route is left to the Jev loop to work out
LINE_GATE = 0.35      # below it the user is offered the likeliest lines instead
TABLE_FLOOR = 0.15    # a table this unlikely is not looked into
YES = 0.5

NONE_TABLE = "none_of_these_tables"
NONE_LINE = "— none of these lines —"

ROUTES = {
    "data": "Look up figures in the data and answer with them, without a chart.",
    "data_chart": "Look up figures in the data and draw a chart of them.",
    "previous_answer": "Present the previous answer differently — another language, layout or "
                       "explanation — without looking up new figures.",
    "definition": "Explain what a metric or term means or how it is calculated.",
    "other": "Small talk, or a request that has nothing to do with the company's data.",
    "complex": "Figures that need reasoning or a calculation beyond reading report lines: a "
               "ranking of many lines, a growth rate, a cause, a forecast.",
}
TREE_ROUTES = ("data", "data_chart")
# Where a route the tree does not take goes next, when the tree is sure of it.
# The previous answer is left to the loop's Jev: drawn, or put another way.
HAND_TO_BY_ROUTE = {"definition": "answer", "other": "answer", "complex": "sql"}

READ_WITH_CONTEXT = "Read `message` together with `conversation`."


@dataclass
class Plan:
    """Where the tree went for one message, and what it decided."""

    route: str = ""
    route_confidence: float = 0.0
    outcome: str = ""               # query | not_in_data | ask | llm
    reason: str = ""
    hand_to: str | None = None      # outcome llm: "sql" — the SQL agent; "answer" — no data needed
    table: str | None = None
    line: str | None = None
    line_confidence: float = 0.0
    breakdown: bool = False
    duplicate_line: bool = False    # the line name occurs on more than one row
    entities: list[str] = field(default_factory=list)
    unlisted_entity: bool = False
    period: str | None = None
    measure: str | None = None
    columns: list[str] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    sql: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)


def _masked(text: str, limit: int = 240) -> str:
    # Figures stay on the machine: every number of two digits or more becomes "#".
    return re.sub(r"\d{2,}", "#", " ".join(str(text).split()))[:limit]


def make_state(message: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    turns = [
        {"role": t.get("role", "user"), "content": t["content"] if t.get("role") == "user" else _masked(t["content"])}
        for t in (history or []) if t.get("content")
    ]
    return {"message": message, "conversation": turns[-6:]}


def _period_label(p: str) -> str:
    return "Year to date: cumulative since January" if p == "ytd" else f"{p.title()} (the month)"


def stage1_questions(options: dict[str, TableOptions]) -> dict[str, Any]:
    periods = [p for p in (*MONTHS, "ytd") if any(p in t.periods for t in options.values())]
    tables = {name: (t.description or name)[:400] for name, t in options.items()}
    tables[NONE_TABLE] = "None of these tables holds what the user asks for."
    return {
        "route": Choice(
            instructions=f"{READ_WITH_CONTEXT} What should a chat assistant over the company's "
                         "financial reports do now? A short reply — a table name, a correction — "
                         "continues the user's earlier request.",
            criteria=ROUTES,
        ),
        "table": Choice(
            instructions=f"{READ_WITH_CONTEXT} Which of these report tables holds the figures the user asks for?",
            criteria=tables,
        ),
        "period": Choice(
            instructions=f"{READ_WITH_CONTEXT} Which period does the user ask about? A year alone is not a period.",
            criteria={**{p: _period_label(p) for p in periods}, "not_stated": "No month and no year-to-date is named."},
        ),
        "measure": Choice(
            instructions=f"{READ_WITH_CONTEXT} Does the user ask for actual figures or for the budget (plan)?",
            criteria={"actual": "Actual figures, the fact", "budget": "Budget or plan figures",
                      "not_stated": "Neither is named"},
        ),
        "all_entities": Noul(
            instructions=f"{READ_WITH_CONTEXT} The user asks for every company or business unit of "
                         "the report — all of them — not for particular ones.",
        ),
        "breakdown": Noul(
            instructions=f"{READ_WITH_CONTEXT} The user asks for the details, the breakdown or the "
                         "components of a report line — the lines it is made of — and not only for "
                         "its total.",
        ),
        "names_company": Noul(
            instructions=f"{READ_WITH_CONTEXT} The message names a particular company, brand or "
                         "business unit — not \"all companies\" and not a line of the report.",
        ),
        "ranking": Noul(
            instructions=f"{READ_WITH_CONTEXT} The user asks for a ranking: the top, largest, smallest, "
                         "best or worst lines or companies. A comparison of companies the user names "
                         "is not a ranking.",
        ),
        "several_lines": Noul(
            instructions=f"{READ_WITH_CONTEXT} The user asks about two or more different lines of the "
                         "report at once, each named on its own — not about one line for several "
                         "companies or periods, and not about the lines one line is made of.",
        ),
    }


def stage2_questions(table: TableOptions) -> dict[str, Any]:
    questions: dict[str, Any] = {
        "line": Choice(
            instructions=f"{READ_WITH_CONTEXT} Which line of this report does the user ask about? A "
                         "line is what is measured — a revenue, a cost, a profit; the names of companies "
                         "are the report's columns, not its lines. The lines are named in English; the "
                         "user may write in Russian or Azerbaijani, in Latin letters, with typos. Choose "
                         "the last option when no line matches.",
            criteria={**{label: table.describe(label) for label in table.labels[:254]},
                      NONE_LINE: "No line of this report matches what the user asks for."},
        ),
    }
    if len(table.entities) > 1:
        for index, entity in enumerate(table.entities):
            questions[f"entity_{index}"] = Noul(instructions={
                "question": f"{READ_WITH_CONTEXT} The user wants figures of `entity` — a company, brand or "
                            "business unit of the report. Its name may be misspelled or transliterated.",
                "entity": entity,
            })
        questions["unlisted_entity"] = Noul(instructions={
            "question": f"{READ_WITH_CONTEXT} The message names a company, brand or business unit that is "
                        "not one of `known_entities`, even allowing for misspellings. The names in "
                        "`report_lines` are lines of the report, not companies.",
            "known_entities": table.entities,
            "report_lines": table.labels[:254],
        })
    return questions


def _ask(client: TypeSafeClient, stage: str, state: dict[str, Any], questions: dict[str, Any], plan: Plan) -> Any:
    with tracing.span(f"jev: {stage}", as_type="generation", model="jev-latest",
                      input={"state": state, "questions": sorted(questions)}) as call:
        started = time.perf_counter()
        response = client.system_one(state, questions)
        ms = (time.perf_counter() - started) * 1000
        tokens = int(getattr(getattr(response, "usage", None), "input_tokens", 0) or 0)
        if call is not None:
            call.update(
                model=response.model, output=response.model_dump(mode="json"),
                usage_details={"input": tokens}, cost_details={"input": tokens * USD_PER_INPUT_TOKEN},
                metadata={"latency_ms": round(ms)},
            )
    plan.calls.append({"stage": stage, "ms": round(ms), "tokens": tokens})
    return response.answers


def plan_question(
    client: TypeSafeClient,
    message: str,
    history: list[dict[str, str]] | None,
    options: dict[str, TableOptions],
) -> Plan:
    """Walks the tree for one message. Nothing is run against the database here."""
    plan = Plan()
    state = make_state(message, history)

    a1 = _ask(client, "route · table · period", state, stage1_questions(options), plan)
    plan.route, plan.route_confidence = a1["route"].choice, a1["route"].confidence
    if plan.route_confidence < ROUTE_GATE:
        plan.outcome, plan.reason = "llm", f"unsure route ({plan.route_confidence:.2f})"
        return plan
    if plan.route not in TREE_ROUTES:
        plan.outcome, plan.reason = "llm", f"route {plan.route}: not a lookup"
        plan.hand_to = HAND_TO_BY_ROUTE.get(plan.route)
        return plan

    period = a1["period"].choice
    plan.period = None if period == "not_stated" else period
    measure = a1["measure"].choice
    plan.measure = None if measure == "not_stated" else measure
    plan.breakdown = a1["breakdown"].noul >= YES
    all_entities = a1["all_entities"].noul >= YES
    names_company = a1["names_company"].noul >= YES
    if a1["ranking"].noul >= YES:
        plan.outcome, plan.reason, plan.hand_to = "llm", "a ranking: not a lookup of named lines", "sql"
        return plan
    if a1["several_lines"].noul >= YES:
        plan.outcome, plan.reason, plan.hand_to = "llm", "more than one line: the tree takes one", "sql"
        return plan

    table_probs = sorted(
        ((name, p) for name, p in a1["table"].probabilities.items() if name in options),
        key=lambda kv: -kv[1],
    )
    candidates = [name for name, p in table_probs[:2] if p >= TABLE_FLOOR] or [table_probs[0][0]]

    # The candidate tables are looked into at the same time; each thread gets its
    # own copy of the context, so the spans still nest under this message's trace.
    def look_into(name: str) -> tuple[str, Any]:
        return name, _ask(client, f"line · entities in {name}", state, stage2_questions(options[name]), plan)

    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        futures = [pool.submit(contextvars.copy_context().run, look_into, name) for name in candidates]
        looked = dict(f.result() for f in futures)

    # The paths, best first: each looked-into table with its likeliest line.
    p_table = dict(table_probs)
    paths = sorted(
        ((p_table.get(name, 0) * max((p for label, p in a["line"].probabilities.items() if label != NONE_LINE), default=0),
          name) for name, a in looked.items()),
        reverse=True,
    )
    words = user_words(message, history)
    # Companies the user wrote, in any of the tables looked into: a table that
    # has none of them cannot answer, however likely it looked.
    named_anywhere = {e for name in looked for e in spelled_entities(options[name], words)}

    first: Plan | None = None
    for _, name in paths:
        attempt = _resolve(plan, name, options[name], looked[name], message, words,
                           all_entities=all_entities, names_company=names_company,
                           named_anywhere=named_anywhere)
        if attempt.outcome in ("query", "ask"):
            return attempt
        first = first or attempt
    return first or plan


def spelled_entities(table: TableOptions, words: list[str]) -> set[str]:
    own, _ = spelling(table.entities, words)
    return {e for e, score in own.items() if score >= CLOSE and len(name_key(e) or "") >= 4}


def _resolve(
    base: Plan, name: str, table: TableOptions, answers: Any, message: str, words: list[str],
    *, all_entities: bool, names_company: bool, named_anywhere: set[str],
) -> Plan:
    """One table's path to the end: its line, its companies, its columns."""
    plan = replace(base, calls=base.calls, table=name)
    line = answers["line"]
    if line.choice == NONE_LINE:
        plan.outcome, plan.reason = "not_in_data", f"no line of {name} matches"
        return plan
    plan.line, plan.line_confidence = line.choice, line.confidence
    plan.line = prefer_cost_line(message, plan.line, line.probabilities)
    plan.alternatives = [label for label, _ in sorted(line.probabilities.items(), key=lambda kv: -kv[1])
                         if label != NONE_LINE][:3]
    plan.duplicate_line = plan.line in table.duplicates

    if len(table.entities) == 1:
        if named_anywhere and not (named_anywhere & set(table.entities)):
            plan.outcome, plan.reason = "not_in_data", f"{name} has none of the companies asked about"
            return plan
        if names_company and not named_anywhere:
            # A table without companies must not answer for one the data does not have.
            plan.outcome, plan.reason = "not_in_data", "the company asked about is not in the data"
            return plan
        plan.entities = list(table.entities)
    else:
        jev_yes = {e: answers[f"entity_{i}"].noul >= YES for i, e in enumerate(table.entities)}
        named, dropped = reconcile(table.entities, jev_yes, words)
        plan.unlisted_entity = dropped or answers["unlisted_entity"].noul >= YES
        if named:
            plan.entities = named
        elif all_entities:
            plan.entities = list(table.entities)
        elif named_anywhere:
            plan.outcome, plan.reason = "not_in_data", f"{name} has none of the companies asked about"
            return plan
        elif names_company:
            plan.unlisted_entity = True             # a company was named, and it is none of these
        elif not plan.unlisted_entity:
            plan.entities = list(table.entities)    # no company named: every one the table has
        if not plan.entities:
            plan.outcome, plan.reason = "not_in_data", "the company asked about is not in the data"
            return plan

    if plan.line_confidence < LINE_GATE:
        plan.outcome, plan.reason = "ask", f"unsure line ({plan.line_confidence:.2f})"
        return plan
    columns = table.pick_columns(plan.entities, plan.period, plan.measure)
    if not columns:
        plan.outcome, plan.reason = "not_in_data", f"{name}: no column for this period and measure"
        return plan
    plan.columns = [c.column for c in columns]
    plan.sql = build_sql(plan, table, columns)
    plan.outcome = "query"
    return plan


# Generic report vocabulary, not names from the data: the user asks about a cost,
# and the line picked is a revenue line with nearly the same name ("Maintenance"
# for "Maintenance cost") — then a cost line among the likely ones wins.
COST_WORDS = ("cost", "expens", "spend", "spent", "расход", "затрат", "потрат", "xərc", "xerc")
COST_LINE = re.compile(r"\b(cost|costs|expenses?|fees?|cos)\b", re.IGNORECASE)


def prefer_cost_line(message: str, line: str, probabilities: dict[str, float], top: int = 5) -> str:
    words = message.lower().split()
    if not any(w.startswith(c) for w in words for c in COST_WORDS) or COST_LINE.search(line):
        return line
    likely = sorted((p, label) for label, p in probabilities.items() if label != NONE_LINE)[-top:]
    costs = [label for p, label in reversed(likely) if COST_LINE.search(label) and p >= 0.02]
    return costs[0] if costs else line


def _alias(c: Any, *, entity: bool = True) -> str:
    period = "YTD" if c.period == "ytd" else c.period.title()
    parts = (c.entity.title() if entity else "", period, c.measure.title(), str(c.year or ""))
    return " ".join(part for part in parts if part)


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


COMPANY = "Company"


def build_sql(plan: Plan, table: TableOptions, columns: list[Any]) -> str:
    """The query, from the chosen options only: names come from the passport."""
    label = quote_identifier(table.label_column)
    if plan.breakdown and table.level_column:
        where = f"{quote_identifier(table.level_column)} = {_literal(plan.line)}"
    else:
        where = f"{label} = {_literal(plan.line)}"
    entities = list(dict.fromkeys(c.entity for c in columns))
    if len(entities) >= 2 and not plan.breakdown and not plan.duplicate_line:
        return _by_company_sql(table, columns, entities, where)
    select = [f'{label} AS "Line"'] + [f"{quote_identifier(c.column)} AS {quote_identifier(_alias(c))}" for c in columns]
    return f"SELECT {', '.join(select)} FROM {quote_identifier(table.name)} WHERE {where}"


def _by_company_sql(table: TableOptions, columns: list[Any], entities: list[str], where: str) -> str:
    """One line for several companies: a row per company, its periods as the
    columns, the largest first. A company with only 0 or nothing for the line
    is left out; the text says how many were."""
    heads = list(dict.fromkeys(_alias(c, entity=False) for c in columns))
    parts = []
    for entity in entities:
        own = {_alias(c, entity=False): c for c in columns if c.entity == entity}
        values = [f"{quote_identifier(own[h].column) if h in own else 'NULL'} AS {quote_identifier(h)}" for h in heads]
        parts.append(f"SELECT {_literal(entity.title())} AS {quote_identifier(COMPANY)}, {', '.join(values)} "
                     f"FROM {quote_identifier(table.name)} WHERE {where}")
    nonzero = " OR ".join(f"COALESCE({quote_identifier(h)}, 0) != 0" for h in heads)
    return f"SELECT * FROM ({' UNION ALL '.join(parts)}) WHERE {nonzero} ORDER BY 2 DESC"
