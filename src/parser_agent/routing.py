"""Model routing: which LLM serves which stage of the pipeline.

The only place where the default models are defined is DEFAULT_ROUTES below.
To bind a model to a new stage: add a value to Task and a line to
DEFAULT_ROUTES. Everything else (CLI, environment variables, .env) picks the
task up automatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Mapping


class Task(StrEnum):
    """Pipeline stages that need an LLM.

    Stages 1 (compress) and 3 (extract) are pure Python, require no models and
    are not listed here.
    """

    DESCRIBE = "describe"          # Stage 2: compressed sheet -> JSON schema
    CATALOG = "catalog"            # Stage 4: table facts -> catalog descriptions
    ORCHESTRATOR = "orchestrator"  # The chat: the answer analyst of the Jev loop
    SQL = "sql"                    # The chat crew: the data retrieval analyst
    VIZ = "viz"                    # The chat crew: the visualization specialist
    # Placeholder for future stages, for example:
    # VERIFY = "verify"        # re-checking the schema with a stronger model
    # REPAIR = "repair"        # repairing rows that failed type coercion


# Default models for each task.
DEFAULT_ROUTES: dict[Task, str] = {
    Task.DESCRIBE: "gpt-4o-mini",
    Task.CATALOG: "gpt-4o-mini",
    Task.ORCHESTRATOR: "gpt-4.1",
    Task.SQL: "gpt-4o",
    Task.VIZ: "gpt-4.1",
}

# Environment variables (they also work through .env):
#   PARSER_AGENT_MODEL           — for all tasks at once
#   PARSER_AGENT_MODEL_DESCRIBE  — only for the describe task
#   PARSER_AGENT_MODEL_CATALOG   — only for the catalog task
#   PARSER_AGENT_MODEL_ORCHESTRATOR / _SQL / _VIZ — the agents of the chat crew
ENV_PREFIX = "PARSER_AGENT_MODEL"
GLOBAL_ENV_VAR = ENV_PREFIX


def env_var_name(task: Task) -> str:
    """Name of the environment variable that sets the model for a task."""
    return f"{ENV_PREFIX}_{task.value.upper()}"


def parse_task(value: str) -> Task:
    """String -> Task with a clear error on a typo."""
    try:
        return Task(value.strip().lower())
    except ValueError as exc:
        known = ", ".join(t.value for t in Task)
        raise ValueError(f"Неизвестная задача {value!r}. Доступные: {known}") from exc


def parse_override(spec: str) -> tuple[Task | None, str]:
    """Parses the value of the --model flag.

    "gpt-4o"          -> (None, "gpt-4o")           — for all tasks
    "describe=gpt-4o" -> (Task.DESCRIBE, "gpt-4o")  — only for describe
    """
    text = spec.strip()
    if not text:
        raise ValueError("Пустое значение --model")
    if "=" not in text:
        return None, text
    task_part, _, model_part = text.partition("=")
    model = model_part.strip()
    if not model:
        raise ValueError(f"Не указана модель в {spec!r} (ожидается задача=модель)")
    return parse_task(task_part), model


@dataclass(frozen=True)
class ModelRouter:
    """The resolved «task -> model» mapping plus the source of each value."""

    models: dict[Task, str]
    sources: dict[Task, str]   # for the `parser_agent models` output

    def model_for(self, task: Task) -> str:
        return self.models[task]

    def table(self) -> list[tuple[str, str, str]]:
        """[(task, model, where it came from)] in the declaration order of Task."""
        return [(task.value, self.models[task], self.sources[task]) for task in Task]


def build_router(
    overrides: Iterable[str] = (),
    env: Mapping[str, str] | None = None,
) -> ModelRouter:
    """Builds the router honouring the priority order.

    From highest to lowest:
      1. --model task=model         (targeted override from the CLI)
      2. --model model              (global override from the CLI)
      3. PARSER_AGENT_MODEL_<TASK>  (targeted, from the environment or .env)
      4. PARSER_AGENT_MODEL         (global, from the environment or .env)
      5. DEFAULT_ROUTES             (default values)

    Flags take priority over the environment: what is explicitly typed in the
    command must not be overridden by a setting from .env.
    """
    env = os.environ if env is None else env

    global_cli: str | None = None
    per_task_cli: dict[Task, str] = {}
    for spec in overrides or ():
        task, model = parse_override(spec)
        if task is None:
            global_cli = model
        else:
            per_task_cli[task] = model

    global_env = (env.get(GLOBAL_ENV_VAR) or "").strip() or None

    models: dict[Task, str] = {}
    sources: dict[Task, str] = {}
    for task in Task:
        task_env = (env.get(env_var_name(task)) or "").strip() or None
        if task in per_task_cli:
            models[task], sources[task] = per_task_cli[task], f"--model {task.value}="
        elif global_cli is not None:
            models[task], sources[task] = global_cli, "--model"
        elif task_env is not None:
            models[task], sources[task] = task_env, env_var_name(task)
        elif global_env is not None:
            models[task], sources[task] = global_env, GLOBAL_ENV_VAR
        else:
            models[task], sources[task] = DEFAULT_ROUTES[task], "по умолчанию"

    return ModelRouter(models=models, sources=sources)


def resolve_model(task: Task, env: Mapping[str, str] | None = None) -> str:
    """The model for a single task, honouring the environment (no CLI overrides)."""
    return build_router(env=env).model_for(task)
