"""Writes the secrets of the local Langfuse stack and connects the project to it.

    python3 langfuse/init_env.py

1. langfuse/.env — the passwords of Postgres, ClickHouse, Redis and MinIO, the
   encryption keys, and the organisation, project, API keys and user that
   Langfuse creates on its first start (headless initialisation). The file is
   written once and never overwritten: the data volumes are encrypted and locked
   with these values, and new ones would lock the stack out of its own data.
2. The project .env — LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and
   LANGFUSE_BASE_URL are appended if they are not there yet. Existing lines,
   the OpenAI key among them, are left exactly as they are.
"""

from __future__ import annotations

import secrets
from pathlib import Path

HERE = Path(__file__).resolve().parent
STACK_ENV = HERE / ".env"
PROJECT_ENV = HERE.parent / ".env"

BASE_URL = "http://localhost:3000"
USER_EMAIL = "admin@parser-agent.local"


def stack_values() -> dict[str, str]:
    postgres_password = secrets.token_hex(16)
    minio_password = secrets.token_hex(16)
    return {
        "NEXTAUTH_URL": BASE_URL,
        "NEXTAUTH_SECRET": secrets.token_hex(32),
        "SALT": secrets.token_hex(32),
        "ENCRYPTION_KEY": secrets.token_hex(32),     # 64 hex characters, as Langfuse requires
        "POSTGRES_PASSWORD": postgres_password,
        "DATABASE_URL": f"postgresql://postgres:{postgres_password}@postgres:5432/postgres",
        "CLICKHOUSE_PASSWORD": secrets.token_hex(16),
        "REDIS_AUTH": secrets.token_hex(16),
        "MINIO_ROOT_PASSWORD": minio_password,
        "LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY": minio_password,
        "LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY": minio_password,
        "LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY": minio_password,
        # Anonymous usage statistics of the Langfuse server: off, the stack is local.
        "TELEMETRY_ENABLED": "false",
        "LANGFUSE_INIT_ORG_ID": "parser-agent",
        "LANGFUSE_INIT_ORG_NAME": "Parser Agent",
        "LANGFUSE_INIT_PROJECT_ID": "langfuse-parser",
        "LANGFUSE_INIT_PROJECT_NAME": "langfuse_parser",
        "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": f"pk-lf-{secrets.token_hex(16)}",
        "LANGFUSE_INIT_PROJECT_SECRET_KEY": f"sk-lf-{secrets.token_hex(16)}",
        "LANGFUSE_INIT_USER_EMAIL": USER_EMAIL,
        "LANGFUSE_INIT_USER_NAME": "admin",
        "LANGFUSE_INIT_USER_PASSWORD": secrets.token_urlsafe(12),
    }


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


def main() -> None:
    if STACK_ENV.exists():
        print(f"{STACK_ENV} уже есть — оставляю как есть")
    else:
        lines = [f"{key}={value}" for key, value in stack_values().items()]
        STACK_ENV.write_text(
            "# Secrets of the local Langfuse stack — written once by init_env.py.\n"
            + "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        STACK_ENV.chmod(0o600)
        print(f"{STACK_ENV} создан")

    stack = read_env(STACK_ENV)
    wanted = {
        "LANGFUSE_PUBLIC_KEY": stack["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"],
        "LANGFUSE_SECRET_KEY": stack["LANGFUSE_INIT_PROJECT_SECRET_KEY"],
        "LANGFUSE_BASE_URL": stack["NEXTAUTH_URL"],
    }
    present = read_env(PROJECT_ENV)
    missing = {key: value for key, value in wanted.items() if key not in present}
    if missing:
        text = PROJECT_ENV.read_text(encoding="utf-8") if PROJECT_ENV.exists() else ""
        prefix = "" if not text or text.endswith("\n") else "\n"
        with PROJECT_ENV.open("a", encoding="utf-8") as handle:
            handle.write(prefix + "".join(f"{key}={value}\n" for key, value in missing.items()))
        print(f"в {PROJECT_ENV} добавлено: {', '.join(missing)}")
    else:
        print(f"в {PROJECT_ENV} подключение к Langfuse уже есть")


if __name__ == "__main__":
    main()
