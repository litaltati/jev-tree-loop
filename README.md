# Parser Agent: Jev tree + Jev loop

Turns Excel management reports into a SQLite database and answers questions
about it in a web chat. In this variant the chat is routed by
[Jev](https://docs.typesafe.ai), TypeSafe's decision model, instead of a CrewAI
coordinator. The whole app runs as one [CrewAI](https://crewai.com) Flow and is
traced to a local [Langfuse](https://langfuse.com).

```
question ─▶ Jev, stage 1: what does it need? ─┬─ one line of the report ─▶ Jev tree, stage 2: line, companies ─▶ SQL
                                              ├─ more (several lines, a ranking, a calculation) ─▶ SQL agent
                                              └─ no data (a definition, small talk) ──────────────────────┐
                                                        │                                                 │
                                    rows in one shared state                                              │
                                                        ▼                                                 │
                                    Jev again: a chart wanted? ── yes ─▶ visualizer: chart id in the state │
                                                        │ no                         │                    │
                                                        ▼                            ▼                    │
                                    answer analyst: the rows, the chart, the notes ─▶ the answer ◀─────────┘
```

- **Stage 1** (`src/parser_agent/jev_tree/tree.py`). One call to Jev answers
  every question of the plan at once: the route, the table, the period, actual
  or budget, all companies or named ones, a breakdown, a ranking, more than one
  line. One line of the report — for one company or many, one period or all —
  goes to the tree. More than one line, a ranking or a calculation goes to the
  SQL agent. A definition or small talk needs no data.
- **Stage 2, the Jev tree.** Jev picks the line and the companies of the one or
  two likeliest tables; each choice is one of the options read from the table
  passports. Code checks the company names by spelling and builds the SQL. A
  line for several companies comes back as a row per company, the largest
  first; companies with only 0 for it are left out and counted. When a company
  or a period is not in the data, the analyst says so. A word spelled like a
  table's name ("конт 3") is never taken for a missing company: the SQL agent
  takes the question. The tree writes no answer and draws no chart itself.
- **One shared state** (`src/parser_agent/jev_loop/state.py`): the question, the
  conversation, what the tree noted, the rows, the chart id, the steps. Every
  agent reads it and adds to it.
- **Jev again** (`router.py`), with the state as it is by then. It picks the
  next step among those code still allows:

  | Step | Agent | Default model |
  |---|---|---|
  | sql | Data Retrieval Analyst finds the table (`catalog_search`) and runs SQL (`sql_query`) | gpt-4o |
  | visualizer | Data Visualization Specialist draws a chart in a Docker sandbox | gpt-4.1 |
  | exit | Answer Analyst writes the answer from the state | gpt-4.1 |

  Code limits what Jev may choose: no new SQL once the tree found the rows (its
  lookup is checked by code), at most two SQL runs, a chart only when there are
  rows, nothing but the answer after the chart, at most four steps. When only
  one step is left, Jev is not asked. Jev sees a description of the rows
  (table, columns, row count), never the figures. A chart is kept as a PNG
  file; the state holds its id.
- **Without Jev.** If `TYPESAFE_API_KEY` is missing or Jev cannot be reached,
  code takes the steps: the SQL agent once, then the answer.

Under every answer the page says who found the figures: "Figures found by the
Jev tree" or "Answered by the Jev loop" (the SQL agent, or no data).

**Parsing** is the same as in the original Parser Agent. Each sheet becomes one
table. The workbook is converted (Aspose.Cells for `.xlsb/.xls/.xlsm/.ods`) and
compressed before any model sees it. The model describes the schema, code loads
the rows, and every table gets a *passport*: its columns, types, samples and the
values that occur.

## Requirements

| | |
|---|---|
| Python 3.11–3.13 | https://www.python.org/downloads/ |
| Docker Desktop: the chart sandbox, plus Langfuse and Qdrant if you use them | https://www.docker.com/products/docker-desktop/ |
| An OpenAI API key: the loop's agents and the Excel parsing | https://platform.openai.com/api-keys |
| A TypeSafe API key: Jev | https://console.typesafe.ai/keys |
| An Excel report, .xlsx or .xlsb | |

No data, keys or accounts come with the repository. You load your own report
and create your own login.

## Run it (macOS / Linux)

```bash
git clone <this repository's URL> && cd <its folder>
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env            # put OPENAI_API_KEY and TYPESAFE_API_KEY in it

python -m parser_agent sandbox build     # once: the Docker image that draws the charts
python -m parser_agent sandbox check     # draws a test chart

mkdir -p data && cp /path/to/report.xlsx data/
python -m parser_agent convert data/report.xlsx output.db    # tables → output.db, passports → output/

python -m parser_agent users add tester  # asks for a password
python -m parser_agent serve output.db --port 8601
```

Open http://127.0.0.1:8601 and sign in as `tester`. The page comes in English,
Azerbaijani and Russian. The messages of the command line are in Russian.

On Windows, create the environment with `py -3.13 -m venv .venv`, then run
`.venv\Scripts\activate`. The other commands are the same.

Run `convert` once per file into the same database. A sheet that is already
there is replaced. To check the compression without calling the API, add
`--dry-run`.

### Accounts

The chat opens only behind a login.

```bash
python -m parser_agent users add anna                    # all tables
python -m parser_agent users add viewer --tables sales   # only these tables
python -m parser_agent users access viewer --all         # change access later
python -m parser_agent users list
python -m parser_agent users passwd anna                 # or: users remove anna
```

## Trying it

- **One line.** Ask "<line of the report> of <company> in <month year>", or for
  all companies. The tree finds the rows, the analyst writes the answer, and it
  is marked "Figures found by the Jev tree".
- **More than one line, rankings, calculations.** The SQL agent finds the rows;
  the answer is marked "Answered by the Jev loop".
- **A chart.** Ask for one, or say "draw it" after an answer. After the rows,
  Jev picks the visualizer, then the analyst writes the answer.
- **Definitions, small talk.** The analyst answers without a query.
- **A company that is not in the data.** The analyst says so instead of
  answering with another company's figures.
- **Without the tree.** Add `PARSER_AGENT_JEV_TREE=off` to `.env` and restart
  `serve`: Jev then picks every step, and the SQL agent finds the figures.

## Optional: Langfuse (time, cost and tokens of every answer)

```bash
python3 langfuse/init_env.py                 # writes langfuse/.env, adds the Langfuse keys to .env
cd langfuse && docker compose up -d && cd ..
```

Restart `serve` afterwards. Open http://localhost:3000 and sign in as
`admin@parser-agent.local` with the password from `LANGFUSE_INIT_USER_PASSWORD`
in `langfuse/.env`. Write that file once and keep it: the data volumes are
locked with its keys. The stack needs about 4 GB of memory. `docker compose
down` in `langfuse/` stops it, and `down -v` deletes all traces. Every question
becomes a trace: the tree's Jev calls, the loop's steps with Jev's
probabilities, every model call, the SQL and the tokens.

## Optional: Qdrant (finding tables by meaning)

Without Qdrant, the SQL agent finds tables by the words in their passports.

```bash
docker run -d --name parser-agent-qdrant -p 127.0.0.1:7333:6333 \
  -v parser-agent-qdrant:/qdrant/storage qdrant/qdrant
python -m parser_agent index          # puts the passports from output/ into Qdrant
```

## Settings

| Variable | Default | What it does |
|---|---|---|
| `PARSER_AGENT_JEV_TREE` | on | `off` skips the tree: Jev picks every step, the SQL agent finds the figures |
| `PARSER_AGENT_MODEL_ORCHESTRATOR` | gpt-4.1 | the answer analyst |
| `PARSER_AGENT_MODEL_SQL` | gpt-4o | the SQL agent |
| `PARSER_AGENT_MODEL_VIZ` | gpt-4.1 | the visualizer |
| `PARSER_AGENT_MODEL_DESCRIBE` / `_CATALOG` | gpt-4o-mini | parsing: sheet schema, passports |

`PARSER_AGENT_MODEL` sets every model at once. Run `python -m parser_agent models`
to see which models are in use.

## Ports

| | |
|---|---|
| 8601 | the chat (`serve --port`; 8600 if the option is left out) |
| 3000 | Langfuse |
| 7333 | Qdrant |

## Security

- **Accounts.** Passwords are hashed with scrypt. Sessions use HttpOnly,
  SameSite=strict cookies. An account is locked for 5 minutes after 5 failed
  logins.
- **Table access.** The database enforces each account's tables on every query
  through a SQLite authorizer; the prompt is not relied on. Queries are
  read-only single `SELECT`s, at most 200 rows.
- **Chart sandbox.** Charts are drawn in a container with no network, a
  read-only root, no capabilities, and memory, CPU and time limits.
- **What goes to Jev.** The question and the conversation, with every number of
  two digits or more in earlier answers replaced by `#`. Jev also gets the
  options read from the passports. Figures from the data are never sent.
- **Never committed** (`.gitignore`): `.env`, `langfuse/.env`, `data/`,
  `output/`, and every `*.db` (the data, `auth.db` with the accounts,
  `chats.db` with the saved chats).

## Project layout

```
src/parser_agent/
  flow.py            the Flow: router, parse stages, chat entry (the tree, then the loop)
  jev_tree/          the Jev tree: options from passports, the two stages, name checks, the rows
  jev_loop/          the Jev loop: shared state, Jev's next step, the agents, the loop
  crew/              agents' instructions, tools, progress steps, chart sandbox (Dockerfile)
  compressor/        Aspose conversion and sheet compression (no LLM)
  layout.py          header detection
  describer.py       sheet → schema prompt
  extractor.py       rows into SQLite
  catalog.py         table passports
  indexer.py         passports → Qdrant
  search.py          table search: vectors or words
  chat.py            safe SQL execution, table guard
  auth.py, chats.py  accounts and saved conversations
  web.py, web/       FastAPI server and the chat page
  tracing.py         Langfuse
  routing.py         which model serves which task
  cli.py             the `python -m parser_agent` commands
langfuse/            local Langfuse v4 (docker compose)
```
