"""Stage 2: sending the compressed sheet representation to the LLM and parsing the JSON schema."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable

from pydantic import ValidationError

from .models import SheetSchema, sanitize_table_name
from .routing import Task, resolve_model

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — анализатор структуры Excel-таблиц. Тебе дано сжатое представление одного листа Excel.
Весь лист будет загружен в ОДНУ таблицу SQLite. Твоя задача — определить:

Где находятся строки-заголовки (headers) — одна или несколько строк, которые станут именами колонок SQLite.
Диапазон строк с данными (data_start_row, data_end_row).
Диапазон колонок с данными (data_start_col, data_end_col).
Для каждой колонки: имя (из заголовка), тип данных (TEXT, REAL, INTEGER, DATE).
Если есть иерархия строк (merged cells в первых колонках, вложенные уровни типа "Выручка" → "Продукт А" → "Регион 1"), укажи какие колонки формируют иерархию и сколько уровней.

Верни ТОЛЬКО валидный JSON, без пояснений, без markdown."""

# The response shape is defined separately from the system prompt so that the
# system text stays exactly as it is in the specification.
_JSON_SHAPE = """{
  "sheet_name": "<имя листа>",
  "table_name": "<sanitized имя таблицы>",
  "header_rows": [<номера строк-заголовков>],
  "data_start_row": <int>,
  "data_end_row": <int>,
  "data_start_col": "<буква колонки>",
  "data_end_col": "<буква колонки>",
  "columns": [
    {
      "original_col_letter": "A",
      "name": "<имя колонки>",
      "dtype": "TEXT|REAL|INTEGER|DATE",
      "is_row_label": false,
      "hierarchy_level": null
    }
  ],
  "notes": "<наблюдения о структуре>"
}"""


def build_user_prompt(
    compressed_text: str, sheet_name: str, header_block: str | None = None
) -> str:
    """The user message: the compressed sheet, the header cells and the response shape."""
    headers = ""
    if header_block:
        headers = (
            "\n\nЦепочки заголовков по колонкам. Ячейки шапки уже склеены кодом "
            "сверху вниз через `__` (объединённые ячейки развёрнуты, значения "
            "верхних уровней растянуты на подчинённые им колонки):\n"
            f"{header_block}\n\n"
            "Этот список — ПОЛНЫЙ перечень колонок листа, и columns строится по нему:\n"
            "- ровно одна запись на КАЖДУЮ строку списка, в том же порядке и с той "
            "же буквой в original_col_letter;\n"
            "- не пропускай ни одной колонки, даже если колонки похожи друг на друга "
            "или их нет в сжатом представлении: сжатие намеренно убирает однотипные "
            "колонки, но в таблице нужны все;\n"
            "- тип колонки, которой нет в сжатом представлении, бери по колонкам того "
            "же блока (соседние колонки с той же нижней частью цепочки).\n"
            "Сжатое представление нужно только чтобы понять структуру листа: где "
            "шапка, где данные, какие типы.\n\n"
            "Имя колонки в SQLite — это СОКРАЩЁННАЯ цепочка. Правила:\n"
            "1. Слова и числа бери ТОЛЬКО из цепочки. Ничего не добавляй из "
            "сжатого листа и не выдумывай.\n"
            "2. НЕ ДУБЛИРУЙ информацию. Если верхний уровень повторяет то, что уже "
            "сказано на нижнем (год, месяц, валюта, единица измерения), оставь её "
            "один раз. Верхний уровень шапки часто заполнен небрежно: в нём "
            "повторяют то, что ниже уточняется точнее.\n"
            "3. ГОДЫ — по шагам. Посчитай в цепочке четырёхзначные годы "
            "(19xx/20xx):\n"
            "   - год один — оставь на месте;\n"
            "   - рядом стоит vs — это сравнение, годы не трогай;\n"
            "   - годов два и больше — удали все, кроме ПОСЛЕДНЕГО: он пришёл с "
            "нижнего уровня шапки, ближе к данным, и уточняет верхние.\n"
            "   may_2026__actual_2026 -> may_actual_2026; "
            "may_2026__actual_2025 -> may_actual_2025; "
            "q1_2026__план_2026 -> q1_план_2026; "
            "may_2026__act_vs_bdgt -> may_2026_act_vs_bdgt.\n"
            "4. Обязательно сохрани то, что ОТЛИЧАЕТ колонки друг от друга: имена "
            "колонок должны остаться разными.\n"
            "5. Служебные заглушки внутри цепочки выбрасывай.\n"
            "Если цепочки у колонки нет, назови её по смыслу содержимого."
        )
    return (
        f"Имя листа: {sheet_name}\n\n"
        "Сжатое представление листа (адреса — ОРИГИНАЛЬНЫЕ координаты Excel, "
        "все номера строк и буквы колонок в ответе должны быть в них же):\n"
        f"{compressed_text}"
        f"{headers}\n\n"
        "Верни JSON строго такой формы:\n"
        f"{_JSON_SHAPE}\n\n"
        "Если на листе визуально несколько таблиц — всё равно опиши его как ОДНУ "
        "плоскую таблицу: data_start_row — первая строка данных первого блока, "
        "data_end_row — последняя строка данных последнего блока, а особенности "
        "структуры опиши в notes.\n\n"
        "Обязательно отметь is_row_label=true у колонок, где лежат НАЗВАНИЯ СТРОК "
        "(статьи отчёта, категории — обычно это колонка A). Такие колонки — "
        "заголовки строк, симметричные заголовкам колонок. Уровень вложенности "
        "определяется автоматически по отступам в Excel, поэтому в "
        "hierarchy_level достаточно указать 0 — важен сам флаг is_row_label."
    )


def _strip_code_fence(text: str) -> str:
    """Removes ```json ... ``` around the answer if the model added them anyway."""
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return stripped


def parse_schema(
    raw_text: str,
    sheet_name: str,
    table_name: str | None = None,
) -> SheetSchema:
    """Parses the model answer into a SheetSchema.

    sheet_name and table_name are filled in deterministically: the table name is
    always the sanitized sheet name (the "1 sheet = 1 table" rule), whatever the
    model returned.
    """
    payload: Any = json.loads(_strip_code_fence(raw_text))
    if not isinstance(payload, dict):
        raise ValueError("Ожидался JSON-объект со схемой листа")

    payload["sheet_name"] = sheet_name
    payload["table_name"] = table_name or sanitize_table_name(sheet_name)
    return SheetSchema.model_validate(payload)


def _default_client() -> Any:
    """Lazy openai import: the package is only needed for a real API call."""
    from openai import AsyncOpenAI

    return AsyncOpenAI()


async def describe_sheet(
    compressed_text: str,
    sheet_name: str,
    *,
    model: str | None = None,
    client: Any | None = None,
    table_name: str | None = None,
    header_block: str | None = None,
    on_raw_response: Callable[[str], None] | None = None,
) -> SheetSchema:
    """Sends the compressed text to the LLM and parses the answer into a SheetSchema.

    on_raw_response is called with the model answer BEFORE parsing — as is,
    including a failed attempt before a retry. The CLI uses it to save the
    answer to a file.

    model=None — the model is taken from the routing (routing.py: Task.DESCRIBE,
    honouring PARSER_AGENT_MODEL_DESCRIBE / PARSER_AGENT_MODEL).

    On a failed parse one retry is made: the model is given its own answer and
    the error text, with a request to return corrected JSON.
    """
    model = model or resolve_model(Task.DESCRIBE)
    client = client or _default_client()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(compressed_text, sheet_name, header_block),
        },
    ]

    last_error: Exception | None = None
    for attempt in range(2):  # the first attempt + one retry
        response = await client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
        raw_text = response.choices[0].message.content or ""
        if on_raw_response is not None:
            on_raw_response(raw_text)
        try:
            return parse_schema(raw_text, sheet_name, table_name)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "[%s] ответ модели не разобран (попытка %d): %s", sheet_name, attempt + 1, exc
            )
            if attempt == 0:
                messages.append({"role": "assistant", "content": raw_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Твой ответ не удалось разобрать. Ошибка:\n"
                            f"{exc}\n\n"
                            "Верни ИСПРАВЛЕННЫЙ валидный JSON той же формы. "
                            "Только JSON, без markdown и пояснений."
                        ),
                    }
                )

    raise ValueError(
        f"Не удалось получить валидную схему для листа {sheet_name!r}: {last_error}"
    ) from last_error


def describe_sheet_sync(
    compressed_text: str,
    sheet_name: str,
    *,
    model: str | None = None,
    client: Any | None = None,
    table_name: str | None = None,
) -> SheetSchema:
    """A synchronous wrapper around describe_sheet() for scripts and the REPL."""
    return asyncio.run(
        describe_sheet(
            compressed_text,
            sheet_name,
            model=model,
            client=client,
            table_name=table_name,
        )
    )
