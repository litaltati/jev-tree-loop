"""Accounts of the web interface: users, password hashes, login sessions.

The store is a SQLite file of its own (auth.db by default), apart from the data
databases. Accounts are made by an administrator from the terminal —
`python -m parser_agent users add <name>`; there is no sign-up page.

  * Passwords are kept as scrypt hashes with a random salt. scrypt is slow and
    memory-hard on purpose: a copied auth.db does not give the passwords away.
  * A login gets a random token in an HttpOnly cookie; the server keeps only
    the SHA-256 of it, with an expiry. Logging out deletes it; a new password
    or a removed account ends every session of that user.
  * After MAX_FAILURES wrong passwords in a row an account is locked for
    LOCK_SECONDS — against guessing. An unknown name costs the same time as a
    wrong password, so the answer does not tell which names exist.
  * Usernames are ASCII: Langfuse takes user ids in US-ASCII only, and the
    traces of the chat are grouped by the username.
  * Table access: an account either reads every table, or — restricted — only
    the tables listed for it. The list is enforced where the data is read
    (catalog search and the SQLite authorizer of every query), never by
    telling a model what not to touch.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterator

DEFAULT_AUTH_DB = Path("auth.db")

USERNAME_RE = re.compile(r"[A-Za-z0-9._-]{2,64}")
MIN_PASSWORD_LEN = 8

SESSION_SECONDS = 12 * 3600
MAX_FAILURES = 5
LOCK_SECONDS = 5 * 60

COOKIE_NAME = "pa_session"

# scrypt cost: 2**14 rounds, 16 MiB of memory, about 50 ms per check.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1
_SCRYPT_MAXMEM = 64 * 1024 * 1024

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    failures      INTEGER NOT NULL DEFAULT 0,
    locked_until  REAL NOT NULL DEFAULT 0,
    restricted    INTEGER NOT NULL DEFAULT 0   -- 1: only the tables in user_tables
);
CREATE TABLE IF NOT EXISTS user_tables (
    username   TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
    table_name TEXT NOT NULL,
    PRIMARY KEY (username, table_name)
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    username   TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
    expires_at REAL NOT NULL
);
"""

WRONG_CREDENTIALS = "Wrong username or password"


class AuthError(ValueError):
    """A refusal that can be shown to the person as is."""


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def hash_password(password: str) -> str:
    """`scrypt$n$r$p$salt$hash` — the parameters travel with the hash."""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
        maxmem=_SCRYPT_MAXMEM, dklen=32,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        expected = base64.b64decode(digest)
        candidate = hashlib.scrypt(
            password.encode("utf-8"), salt=base64.b64decode(salt), n=int(n), r=int(r),
            p=int(p), maxmem=_SCRYPT_MAXMEM, dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return scheme == "scrypt" and hmac.compare_digest(candidate, expected)


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    """Checked against when the name is unknown, to spend the same time."""
    return hash_password(secrets.token_hex(16))


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def check_username(username: str) -> str:
    username = (username or "").strip()
    if not USERNAME_RE.fullmatch(username):
        raise AuthError(
            "Имя пользователя: 2–64 символа, латинские буквы, цифры, точка, _ или -"
        )
    return username


def check_password(password: str) -> str:
    if len(password or "") < MIN_PASSWORD_LEN:
        raise AuthError(f"Пароль должен быть не короче {MIN_PASSWORD_LEN} символов")
    return password


class AuthStore:
    """Users and login sessions in one SQLite file."""

    def __init__(self, path: Path = DEFAULT_AUTH_DB) -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        created = not self.path.exists()
        connection = sqlite3.connect(self.path)
        if created:
            os.chmod(self.path, 0o600)   # password hashes: readable by the owner only
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(_SCHEMA)
        # A store made before table access existed: its accounts keep reading every table.
        columns = {row[1] for row in connection.execute("PRAGMA table_info(users)")}
        if "restricted" not in columns:
            connection.execute("ALTER TABLE users ADD COLUMN restricted INTEGER NOT NULL DEFAULT 0")
            connection.commit()
        return connection

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """A connection that is always closed. A refusal (AuthError) still commits:
        the failed attempt it counted must survive the refusal itself."""
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except AuthError:
            connection.commit()
            raise
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    # --- administration (the `users` commands) ---------------------------------

    def add_user(self, username: str, password: str) -> None:
        username, password = check_username(username), check_password(password)
        with self._db() as connection:
            try:
                connection.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, hash_password(password), datetime.now().isoformat(timespec="seconds")),
                )
            except sqlite3.IntegrityError as exc:
                raise AuthError(f"Пользователь {username} уже есть") from exc

    def set_password(self, username: str, password: str) -> None:
        """A new password; every open session of the user ends."""
        password = check_password(password)
        with self._db() as connection:
            updated = connection.execute(
                "UPDATE users SET password_hash = ?, failures = 0, locked_until = 0 "
                "WHERE username = ?",
                (hash_password(password), username),
            ).rowcount
            if not updated:
                raise AuthError(f"Пользователя {username} нет")
            connection.execute("DELETE FROM sessions WHERE username = ?", (username,))

    def remove_user(self, username: str) -> bool:
        with self._db() as connection:
            return bool(
                connection.execute("DELETE FROM users WHERE username = ?", (username,)).rowcount
            )

    def list_users(self) -> list[tuple[str, str, list[str] | None]]:
        """(username, created_at, tables) — tables is None for an account that reads all."""
        with self._db() as connection:
            rows = connection.execute(
                "SELECT username, created_at, restricted FROM users ORDER BY username"
            ).fetchall()
            grants: dict[str, list[str]] = {}
            for username, table in connection.execute(
                "SELECT username, table_name FROM user_tables ORDER BY table_name"
            ):
                grants.setdefault(username, []).append(table)
        return [(name, created, grants.get(name, []) if restricted else None)
                for name, created, restricted in rows]

    def set_access(self, username: str, tables: list[str] | None) -> None:
        """None — every table; a list — only these tables (an empty one: none)."""
        with self._db() as connection:
            if connection.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone() is None:
                raise AuthError(f"Пользователя {username} нет")
            connection.execute("DELETE FROM user_tables WHERE username = ?", (username,))
            connection.execute(
                "UPDATE users SET restricted = ? WHERE username = ?",
                (0 if tables is None else 1, username),
            )
            for table in sorted(set(tables or [])):
                connection.execute(
                    "INSERT INTO user_tables (username, table_name) VALUES (?, ?)", (username, table)
                )

    def allowed_tables(self, username: str) -> frozenset[str] | None:
        """The tables the user may read; None — all of them. An unknown name reads nothing."""
        with self._db() as connection:
            row = connection.execute(
                "SELECT restricted FROM users WHERE username = ?", (username,)
            ).fetchone()
            if row is None:
                return frozenset()
            if not row[0]:
                return None
            return frozenset(
                name for (name,) in connection.execute(
                    "SELECT table_name FROM user_tables WHERE username = ?", (username,)
                )
            )

    def has_users(self) -> bool:
        if not self.path.exists():
            return False
        with self._db() as connection:
            return connection.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None

    # --- the web interface -------------------------------------------------------

    def login(self, username: str, password: str, now: float | None = None) -> str:
        """Checks the password and opens a session. Returns the cookie token."""
        now = time.time() if now is None else now
        with self._db() as connection:
            row = connection.execute(
                "SELECT password_hash, failures, locked_until FROM users WHERE username = ?",
                ((username or "").strip(),),
            ).fetchone()
            if row is None:
                verify_password(password or "", _dummy_hash())
                raise AuthError(WRONG_CREDENTIALS)

            password_hash, failures, locked_until = row
            if locked_until > now:
                minutes = int((locked_until - now) // 60) + 1
                raise AuthError(f"Too many failed attempts — try again in {minutes} min")

            if not verify_password(password or "", password_hash):
                failures += 1
                lock = now + LOCK_SECONDS if failures >= MAX_FAILURES else 0
                connection.execute(
                    "UPDATE users SET failures = ?, locked_until = ? WHERE username = ?",
                    (0 if lock else failures, lock, username.strip()),
                )
                raise AuthError(WRONG_CREDENTIALS)

            token = secrets.token_urlsafe(32)
            connection.execute(
                "UPDATE users SET failures = 0, locked_until = 0 WHERE username = ?",
                (username.strip(),),
            )
            connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            connection.execute(
                "INSERT INTO sessions (token_hash, username, expires_at) VALUES (?, ?, ?)",
                (_token_hash(token), username.strip(), now + SESSION_SECONDS),
            )
            return token

    def user_for(self, token: str | None, now: float | None = None) -> str | None:
        """The user a cookie token belongs to, while its session lasts."""
        if not token:
            return None
        now = time.time() if now is None else now
        with self._db() as connection:
            row = connection.execute(
                "SELECT username FROM sessions WHERE token_hash = ? AND expires_at > ?",
                (_token_hash(token), now),
            ).fetchone()
        return row[0] if row else None

    def logout(self, token: str | None) -> None:
        if token:
            with self._db() as connection:
                connection.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
