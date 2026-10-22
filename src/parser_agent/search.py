"""Which table a question is about — the search behind the chat's catalog_search tool.

Three kinds of evidence, strongest first:

  * a value the question names literally, found in a passport (evidence.py):
    «EBITDA» is a row label of four tables, and the one where that row actually
    holds figures wins;
  * the meaning of the passports, by vectors in Qdrant;
  * a plain substring over the passports, when Qdrant is not running.

When the leading candidates are too close to call, the result says so, and it
is for the conversation to ask which table is meant — not to guess.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .catalog import connect_existing, list_tables, quote_identifier
from .evidence import collect_evidence, with_values
from .indexer import catalog_files, embed_texts, load_catalog

logger = logging.getLogger(__name__)

# How many search hits come back at once.
SEARCH_LIMIT = 10


def read_catalogs(catalog_dir: Path) -> dict[str, dict[str, Any]]:
    """{table name: its passport}. A file that is not a passport is skipped."""
    catalogs: dict[str, dict[str, Any]] = {}
    for path in catalog_files(catalog_dir):
        try:
            catalog = load_catalog(path)
        except ValueError as exc:
            logger.warning("паспорт пропущен: %s", exc)
            continue
        catalogs[catalog["table_name"]] = catalog
    return catalogs


def row_fill(db_path: Path, table: str, column: str, value: str) -> int | None:
    """Сколько чисел заполнено в строке, которую назвал вопрос.

    Строка может существовать в четырёх таблицах и быть пустой в трёх из них.
    Тогда «ничего не нашлось» — это не ошибка запроса, а правда про выбранную
    таблицу, и узнать это лучше до выбора, а не после.
    """
    if not db_path.exists():
        return None
    connection = connect_existing(db_path, read_only=True)
    connection.text_factory = str
    try:
        if table not in list_tables(connection):
            return None
        row = connection.execute(
            f"SELECT * FROM {quote_identifier(table)} "
            f"WHERE {quote_identifier(column)} = ? LIMIT 1",
            (value,),
        ).fetchone()
        if row is None:
            return None
        return sum(1 for v in row if isinstance(v, (int, float)) and v)
    except sqlite3.Error as exc:
        logger.debug("не удалось посчитать заполненность: %s", exc)
        return None
    finally:
        connection.close()


def with_fill(db_path: Path, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Дополняет кандидатов числом заполненных ячеек и ставит непустые вперёд."""
    for item in facts:
        first = item["values"][0]
        filled = row_fill(db_path, item["table_name"], first["column"], first["value"])
        item["filled"] = filled
        if filled is not None:
            note = f"{filled} figures in this row" if filled else "this row is empty here"
            item["why"] = f"{item['why']} · {note}"
    return sorted(
        facts, key=lambda i: (-(i.get("filled") or 0), -i["score"], i["table_name"])
    )


def _searchable_text(catalog: dict[str, Any]) -> str:
    parts = [catalog.get("table_name", ""), catalog.get("table_description", "")]
    for column in catalog.get("columns") or []:
        parts.append(column.get("column_name", ""))
        parts.append(column.get("column_description", ""))
        parts.extend(column.get("allowed_data") or [])
    return "\n".join(parts).lower()


# Words too common to tell tables apart.
_STOP_WORDS = frozenset(
    "the a an of in on for to and or is are was what which how many much show me "
    "give list by per with from all each".split()
)


def search_by_text(catalogs: dict[str, dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """The fallback search, without vectors: the words of the question in the passports.

    A table scores one point for every word of the question found in its name,
    descriptions, column names or values — `allowed_data` among them: the values
    are not in the vector, and a question like «в какой таблице есть регион
    Barda» is answered exactly here. The whole question found verbatim counts
    most. A question that matches nothing gets every table back, scored 0: with
    no evidence the choice is the reader's, not this function's.
    """
    needle = query.strip().lower()
    words = {w for w in re.findall(r"[\w%]+", needle) if len(w) > 2 and w not in _STOP_WORDS}
    hits: list[dict[str, Any]] = []
    for name, catalog in catalogs.items():
        text = _searchable_text(catalog)
        score = sum(1 for word in words if word in text) + (len(words) if needle and needle in text else 0)
        hits.append(
            {
                "table_name": name,
                "schema_name": catalog.get("schema_name", ""),
                "table_description": catalog.get("table_description", ""),
                "score": float(score),
                "columns": len(catalog.get("columns") or []),
            }
        )
    hits.sort(key=lambda hit: (-hit["score"], hit["table_name"]))
    return hits[:SEARCH_LIMIT]


def search_by_vector(
    query: str, *, url: str, collection: str, embed_model: str
) -> list[dict[str, Any]]:
    """The semantic search: the question is embedded and matched against Qdrant.

    One point per table, so what comes back is tables. Their columns arrive with
    them — the whole passport lies in the payload.
    """
    from qdrant_client import QdrantClient

    client = QdrantClient(url=url)
    # Qdrant is asked first: the embedding is a paid call, and it is wasted when
    # the collection is not there to search.
    client.get_collection(collection)
    vector = embed_texts([query], model=embed_model)[0]
    found = client.query_points(
        collection_name=collection, query=vector, limit=SEARCH_LIMIT * 2, with_payload=True
    ).points
    hits = []
    for point in found:
        payload = point.payload or {}
        hits.append(
            {
                "table_name": payload.get("table_name", ""),
                "schema_name": payload.get("schema_name", ""),
                "table_description": payload.get("table_description", ""),
                "score": round(float(point.score), 4),
                "columns": len(payload.get("columns") or []),
            }
        )
    return hits


# Насколько близкими должны быть два первых кандидата, чтобы не гадать, а
# спросить. Отрыв меньше этой доли — вопрос пользователю.
AMBIGUITY_MARGIN = 0.12


def rank_tables(vector_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Кандидаты-таблицы по смыслу их паспортов, лучший первым."""
    candidates: dict[str, dict[str, Any]] = {}

    def entry(name: str) -> dict[str, Any]:
        return candidates.setdefault(
            name,
            {"table_name": name, "score": 0.0, "why": [], "table_description": "",
             "columns": [], "values": []},
        )

    for hit in vector_hits:
        item = entry(hit["table_name"])
        if hit.get("table_description") and not item["table_description"]:
            item["table_description"] = hit["table_description"]
        if hit["score"] > item["score"]:
            item["score"] = hit["score"]
        item["why"].append({"kind": "table", "score": hit["score"]})

    ranked = sorted(candidates.values(), key=lambda c: -c["score"])
    for item in ranked:
        item["score"] = round(item["score"], 4)
    return ranked


def needs_choice(ranked: list[dict[str, Any]]) -> bool:
    """Стоит ли переспросить, вместо того чтобы взять первую таблицу.

    Листы одной книги описаны похоже, и первое место часто выигрывает случайно.
    Если отрыв от второго меньше AMBIGUITY_MARGIN — выбор за человеком.
    """
    if len(ranked) < 2 or ranked[0]["score"] <= 0:
        return False
    return (ranked[0]["score"] - ranked[1]["score"]) / ranked[0]["score"] < AMBIGUITY_MARGIN


@dataclass
class TableSearch:
    """The candidate tables for a question, the best first."""

    candidates: list[dict[str, Any]] = field(default_factory=list)
    by: str = ""              # "values" | "vectors" | "text" — what decided the order
    ambiguous: bool = False   # the leaders are too close to call
    matched: bool = True      # False: nothing in the question pointed at any table


def find_tables(
    question: str,
    catalogs: dict[str, dict[str, Any]],
    db_path: Path,
    *,
    qdrant_url: str,
    collection: str,
    embed_model: str,
) -> TableSearch:
    """Ranks the tables for a question: named values first, then meaning."""
    # What the question names and the tables really hold: stronger than any
    # likeness of descriptions — the same item lives on four sheets, and the
    # fact of presence decides, not the meaning.
    facts = with_fill(db_path, with_values(collect_evidence(question, catalogs)))
    if facts:
        return TableSearch(candidates=facts, by="values", ambiguous=len(facts) > 1)

    try:
        hits = search_by_vector(
            question, url=qdrant_url, collection=collection, embed_model=embed_model
        )
        # Qdrant holds every table; only the ones this search may show count.
        hits = [hit for hit in hits if hit["table_name"] in catalogs]
        by = "vectors"
    except Exception as exc:
        logger.info("Qdrant недоступен (%s), таблицы ищутся по тексту", exc)
        hits = search_by_text(catalogs, question)
        by = "text"
    ranked = rank_tables(hits)
    matched = by == "vectors" or any(item["score"] > 0 for item in ranked)
    return TableSearch(
        candidates=ranked, by=by, ambiguous=matched and needs_choice(ranked), matched=matched
    )
