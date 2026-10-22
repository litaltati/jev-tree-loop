"""Saved conversations of the web chat: one SQLite file (chats.db by default).

It lives apart from the data databases and from the accounts in auth.db:

  * every conversation belongs to one account, and every read and write is
    filtered by the signed-in user — nobody reaches another person's chats;
  * signing out deletes nothing: a conversation stays until its owner deletes it;
  * a conversation's id is the one its traces carry in Langfuse as the session;
  * the title is the beginning of the first question, until it is renamed;
  * the history is kept as it was answered: a later change of table access
    does not rewrite what the user has already been shown.

What an answer showed — the SQL, the rows, the chart — is stored with it, so a
reopened conversation looks the way it did, and a follow-up such as "draw it"
works from the stored query even after the server restarts.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DEFAULT_CHATS_DB = Path("chats.db")

TITLE_CHARS = 80
MAX_SAVED_ROWS = 200

_CHART_ID = re.compile(r"/api/charts/([0-9a-f]{32})")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    username   TEXT NOT NULL,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS conversations_by_user ON conversations (username, updated_at);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    payload         TEXT,              -- the answer as the page renders it (JSON)
    chart_id        TEXT,              -- the chart the answer showed, if any
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages (conversation_id, id);
CREATE INDEX IF NOT EXISTS messages_by_chart ON messages (chart_id);
"""


class NotYours(PermissionError):
    """The conversation id belongs to another account."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_title(question: str) -> str:
    text = " ".join(question.split())
    return text if len(text) <= TITLE_CHARS else text[: TITLE_CHARS - 1].rstrip() + "…"


def chart_id_of(answer: dict[str, Any]) -> str | None:
    match = _CHART_ID.search(str(answer.get("chart") or ""))
    return match.group(1) if match else None


class ChatStore:
    """Conversations and their messages, by owner."""

    def __init__(self, path: Path = DEFAULT_CHATS_DB) -> None:
        self.path = Path(path)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(_SCHEMA)
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def save_turn(
        self, username: str, conversation_id: str, question: str, answer: dict[str, Any]
    ) -> None:
        """A question and its answer, in one transaction. The first question of a
        conversation creates it; an id that belongs to someone else is refused."""
        now = _now()
        payload = dict(answer)
        payload["rows"] = list(payload.get("rows") or [])[:MAX_SAVED_ROWS]
        payload.pop("usage", None)
        with self._db() as connection:
            row = connection.execute(
                "SELECT username FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO conversations (id, username, title, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (conversation_id, username, make_title(question), now, now),
                )
            elif row[0] != username:
                raise NotYours(conversation_id)
            else:
                connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
                )
            connection.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
                (conversation_id, question, now),
            )
            connection.execute(
                "INSERT INTO messages (conversation_id, role, content, payload, chart_id, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?, ?)",
                (conversation_id, str(answer.get("text") or ""),
                 json.dumps(payload, ensure_ascii=False, default=str), chart_id_of(answer), now),
            )

    def list_conversations(self, username: str) -> list[dict[str, Any]]:
        """The user's conversations, the latest first."""
        with self._db() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "WHERE username = ? ORDER BY updated_at DESC, created_at DESC",
                (username,),
            ).fetchall()
        return [{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows]

    def get(self, username: str, conversation_id: str) -> dict[str, Any] | None:
        """One conversation with its messages; None if it is not the user's."""
        with self._db() as connection:
            head = connection.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ? AND username = ?",
                (conversation_id, username),
            ).fetchone()
            if head is None:
                return None
            messages = connection.execute(
                "SELECT role, content, payload FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return {
            "id": head[0], "title": head[1], "created_at": head[2], "updated_at": head[3],
            "messages": [
                {"role": role, "content": content, "answer": json.loads(payload) if payload else None}
                for role, content, payload in messages
            ],
        }

    def rename(self, username: str, conversation_id: str, title: str) -> bool:
        title = make_title(title)
        if not title:
            return False
        with self._db() as connection:
            return bool(connection.execute(
                "UPDATE conversations SET title = ? WHERE id = ? AND username = ?",
                (title, conversation_id, username),
            ).rowcount)

    def delete(self, username: str, conversation_id: str) -> list[str] | None:
        """Deletes the conversation; returns the charts it showed (None if not the user's)."""
        with self._db() as connection:
            if connection.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND username = ?", (conversation_id, username)
            ).fetchone() is None:
                return None
            charts = [r[0] for r in connection.execute(
                "SELECT chart_id FROM messages WHERE conversation_id = ? AND chart_id IS NOT NULL",
                (conversation_id,),
            )]
            connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return charts

    def last_query(self, username: str, conversation_id: str) -> dict[str, Any] | None:
        """The SQL behind the latest answer of the conversation that ran one."""
        with self._db() as connection:
            rows = connection.execute(
                "SELECT m.payload FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                "WHERE c.id = ? AND c.username = ? AND m.role = 'assistant' ORDER BY m.id DESC",
                (conversation_id, username),
            ).fetchall()
        for (payload,) in rows:
            answer = json.loads(payload) if payload else {}
            if answer.get("sql"):
                return {"sql": answer["sql"], "table": answer.get("table")}
        return None

    def owner(self, conversation_id: str) -> str | None:
        """Whose conversation this is; None while it has not been saved yet."""
        with self._db() as connection:
            row = connection.execute(
                "SELECT username FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
        return row[0] if row else None

    def chart_owner(self, chart_id: str) -> str | None:
        with self._db() as connection:
            row = connection.execute(
                "SELECT c.username FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                "WHERE m.chart_id = ? LIMIT 1",
                (chart_id,),
            ).fetchone()
        return row[0] if row else None

    def charts_in_use(self) -> set[str]:
        with self._db() as connection:
            return {r[0] for r in connection.execute(
                "SELECT DISTINCT chart_id FROM messages WHERE chart_id IS NOT NULL"
            )}
