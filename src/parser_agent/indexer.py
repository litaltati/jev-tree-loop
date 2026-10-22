"""Stage 5: the table passports go into Qdrant — one file, one point.

A point is a vector plus a payload. The payload is the whole passport as it is,
so nothing of it is lost; the vector is built from the prose only — the table
name, its description and the columns with theirs.

`allowed_data` deliberately stays out of the vector. A domain of several hundred
region or tariff names outweighs everything that says what the table is about,
and a search for "revenue by region" would stop finding the table that holds it.
The values are still in the payload, where an exact filter can reach them.

The point id is derived from `schema_name` + `table_name`, so a second run
updates the same point instead of breeding duplicates.

One point per table, and no separate points for the columns: the passport in the
payload already carries every column with its type, sample and value domain. The
search finds the table; what to do with its columns is decided by the SQL written
afterwards, which sees them all anyway.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:7333"
DEFAULT_COLLECTION = "table_catalog"

# The embedding model is deliberately kept out of routing.py: there `--model
# gpt-4o` sets every task at once, and a chat model in place of an embedding one
# would only fail at the API.
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# The vector size a collection must be created with, per model.
EMBED_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# How many texts go in one embedding request.
EMBED_BATCH = 96

# Qdrant accepts an integer or a UUID as a point id — not an arbitrary string.
# A fixed namespace turns "schema.table" into the same UUID every run.
POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

CATALOG_SUFFIX = ".catalog.json"


def catalog_files(source: Path) -> list[Path]:
    """The passports to load: a single file, or every one in a folder."""
    if source.is_dir():
        return sorted(source.glob(f"*{CATALOG_SUFFIX}"))
    return [source] if source.is_file() else []


# What tells a passport from the other JSON files of the same folder. The schema
# of stage 2 also has a `table_name`, and by that alone it would pass for a
# passport — and go into Qdrant as a point with empty column names.
REQUIRED_FIELDS = ("table_name", "schema_name", "columns")


def load_catalog(path: Path) -> dict[str, Any]:
    """Reads one passport and checks that it is a passport at all."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: ожидался JSON-объект")

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(
            f"{path}: это не паспорт таблицы, нет полей {', '.join(missing)}. "
            f"Паспорта делает `parser_agent catalog` — это файлы *{CATALOG_SUFFIX}"
        )
    return payload


def point_id(catalog: dict[str, Any]) -> str:
    """A stable id: the same table always lands in the same point."""
    key = f"{catalog.get('schema_name', '')}.{catalog['table_name']}"
    return str(uuid.uuid5(POINT_NAMESPACE, key))


def embedding_text(catalog: dict[str, Any]) -> str:
    """The text the vector is built from: the names and the prose, no value lists."""
    lines = [f"table: {catalog['table_name']} (schema {catalog.get('schema_name', '')})"]
    description = (catalog.get("table_description") or "").strip()
    if description:
        lines.append(description)

    columns = catalog.get("columns") or []
    if columns:
        lines.append("columns:")
        for column in columns:
            name = column.get("column_name", "")
            dtype = column.get("column_data_type", "")
            text = (column.get("column_description") or "").strip()
            head = f"- {name} ({dtype})" if dtype else f"- {name}"
            lines.append(f"{head}: {text}" if text else head)
    return "\n".join(lines)


def build_point(catalog: dict[str, Any], vector: list[float], source: Path | None = None) -> dict:
    """One Qdrant point: the vector plus the whole passport in the payload."""
    payload = dict(catalog)
    payload["kind"] = "table"
    payload["column_names"] = [
        column.get("column_name", "") for column in catalog.get("columns") or []
    ]
    if source is not None:
        payload["source_file"] = source.name
    return {"id": point_id(catalog), "vector": vector, "payload": payload}


def _openai_client() -> Any:
    """Lazy openai import: the package is only needed for a real API call."""
    from openai import OpenAI

    return OpenAI()


def embed_texts(
    texts: list[str], *, model: str = DEFAULT_EMBED_MODEL, client: Any | None = None
) -> list[list[float]]:
    """Vectors for the texts, in the same order. One request for all of them."""
    if not texts:
        return []
    client = client or _openai_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        chunk = texts[start:start + EMBED_BATCH]
        response = client.embeddings.create(model=model, input=chunk)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors.extend(item.embedding for item in ordered)
    return vectors


def _qdrant_client(url: str, api_key: str | None = None) -> Any:
    """Lazy qdrant-client import: it is only needed for a real upload."""
    from qdrant_client import QdrantClient

    return QdrantClient(url=url, api_key=api_key)


def check_collection(client: Any, collection: str, expected_size: int | None) -> int:
    """Checks that the collection is there and its vector size matches the model.

    The size is fixed when a collection is created and cannot be changed later,
    so a mismatch is reported here — with the command that fixes it — instead of
    as an error from deep inside the upload.
    """
    info = client.get_collection(collection)
    vectors = info.config.params.vectors
    size = getattr(vectors, "size", None)
    if size is None:
        raise ValueError(
            f"Коллекция {collection!r} создана с именованными векторами — "
            "такая раскладка не поддерживается"
        )
    if expected_size is not None and size != expected_size:
        raise ValueError(
            f"Коллекция {collection!r} рассчитана на вектор размера {size}, "
            f"а модель даёт {expected_size}. Пересоздайте коллекцию:\n"
            f'  curl -X DELETE .../collections/{collection}\n'
            f'  curl -X PUT .../collections/{collection} -H "Content-Type: application/json" '
            f'-d \'{{"vectors": {{"size": {expected_size}, "distance": "Cosine"}}}}\''
        )
    return size


def upsert_points(
    points: Iterable[dict],
    *,
    collection: str = DEFAULT_COLLECTION,
    url: str = DEFAULT_URL,
    api_key: str | None = None,
    client: Any | None = None,
) -> int:
    """Writes the points into the collection. Returns how many were written."""
    from qdrant_client.models import PointStruct

    batch = [PointStruct(**point) for point in points]
    if not batch:
        return 0
    client = client or _qdrant_client(url, api_key)
    client.upsert(collection_name=collection, points=batch, wait=True)
    return len(batch)


def index_catalogs(
    source: Path,
    *,
    collection: str = DEFAULT_COLLECTION,
    url: str = DEFAULT_URL,
    api_key: str | None = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    openai_client: Any | None = None,
    qdrant_client: Any | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Loads every passport into Qdrant. Returns the points that were built.

    dry_run builds the texts and stops: no vectors are requested and nothing is
    written, so what would go into the point can be looked at first.
    """
    files = catalog_files(source)
    if not files:
        raise FileNotFoundError(f"В {source} нет файлов *{CATALOG_SUFFIX}")

    catalogs = [(path, load_catalog(path)) for path in files]
    texts = [embedding_text(catalog) for _path, catalog in catalogs]

    if dry_run:
        # Точка собирается той же функцией, что и при настоящей записи, только
        # без вектора: показывать надо ровно то, что уедет в Qdrant.
        return [
            {**build_point(catalog, [], path), "text": text}
            for (path, catalog), text in zip(catalogs, texts)
        ]

    # The collection is checked before the embeddings are paid for: a size
    # mismatch cannot be fixed by uploading anyway.
    client = qdrant_client or _qdrant_client(url, api_key)
    size = check_collection(client, collection, EMBED_DIMENSIONS.get(embed_model))

    vectors = embed_texts(texts, model=embed_model, client=openai_client)
    if vectors and len(vectors[0]) != size:
        raise ValueError(
            f"Модель {embed_model} даёт вектор размера {len(vectors[0])}, "
            f"а коллекция {collection!r} рассчитана на {size}"
        )

    points = [
        build_point(catalog, vector, path)
        for (path, catalog), vector in zip(catalogs, vectors)
    ]
    upsert_points(points, collection=collection, client=client)
    logger.info("В коллекцию %s записано точек: %d", collection, len(points))
    return points
