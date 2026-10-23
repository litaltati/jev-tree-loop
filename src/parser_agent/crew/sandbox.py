"""The chart sandbox: the Visualization Specialist's code runs in Docker, not here.

CodeInterpreterTool is gone from crewai-tools, and the sandboxes that replaced
it are cloud services — the rows of the user's reports would leave the machine.
This is a local one, locked down for code a model wrote:

  * no network (`--network none`): nothing can be sent anywhere;
  * no project files: only a throw-away job folder is mounted, holding the rows
    of the query result, the code and the runner — .env, auth.db and the data
    databases are out of reach;
  * a read-only root, no capabilities, no privilege escalation, an unprivileged
    user, limits on memory, CPU and processes, and a wall-clock timeout;
  * what comes out is checked: one PNG, of bounded size, and nothing else is
    taken from the job folder.

The image (sandbox/Dockerfile) is built once: `python -m parser_agent sandbox build`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMAGE = "parser-agent-chart-sandbox:1"
SANDBOX_DIR = Path(__file__).resolve().parent / "sandbox"

TIMEOUT_SECONDS = 30
DOCKER_CHECK_SECONDS = 5   # a Docker that does not answer in this time counts as absent
MEMORY = "512m"
MAX_PNG_BYTES = 5 * 1024 * 1024
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@dataclass
class ChartResult:
    chart_id: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.chart_id is not None


def image_ready() -> bool:
    """Whether the image is there. A Docker that hangs (Docker Desktop not started
    or stuck) or is not installed counts as no image: neither the server's start
    nor an answer may wait on it."""
    try:
        result = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True,
                                timeout=DOCKER_CHECK_SECONDS)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return result.returncode == 0


def build_image() -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "build", "-t", IMAGE, str(SANDBOX_DIR)])


def _docker_command(job: Path, name: str) -> list[str]:
    return [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--read-only", "--tmpfs", "/tmp:rw,size=64m",
        "--memory", MEMORY, "--memory-swap", MEMORY, "--cpus", "1", "--pids-limit", "64",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "-v", f"{job}:/job:rw",
        IMAGE, "python", "/job/runner.py",
    ]


def run_chart_code(
    code: str, columns: list[str], rows: list[list[Any]], charts_dir: Path
) -> ChartResult:
    """Runs the code over the rows; on success the PNG is in charts_dir/<id>.png."""
    if not image_ready():
        return ChartResult(
            error="The chart sandbox is not available: Docker does not answer or the image "
            "is not built. The administrator has to start Docker or run: "
            "python3 -m parser_agent sandbox build"
        )

    with tempfile.TemporaryDirectory(prefix="pa_chart_") as folder:
        job = Path(folder)
        (job / "out").mkdir()
        (job / "out").chmod(0o777)   # the container user is not the owner of the folder
        (job / "data.json").write_text(
            json.dumps({"columns": columns, "rows": rows}, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        (job / "code.py").write_text(code, encoding="utf-8")
        shutil.copy(SANDBOX_DIR / "runner.py", job / "runner.py")

        name = f"pa-chart-{uuid.uuid4().hex[:12]}"
        try:
            proc = subprocess.run(
                _docker_command(job, name), capture_output=True, text=True, timeout=TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return ChartResult(error=f"The code ran longer than {TIMEOUT_SECONDS} s and was stopped")

        if proc.returncode != 0:
            message = (proc.stderr or proc.stdout or "").strip()
            return ChartResult(error=message[-2000:] or f"exit code {proc.returncode}")

        png = job / "out" / "chart.png"
        if not png.is_file() or png.is_symlink():
            return ChartResult(error="No chart was saved to OUT")
        size = png.stat().st_size
        if size > MAX_PNG_BYTES:
            return ChartResult(error=f"The chart is too large ({size} bytes)")
        with png.open("rb") as handle:
            if handle.read(len(PNG_MAGIC)) != PNG_MAGIC:
                return ChartResult(error="OUT is not a PNG image")

        charts_dir.mkdir(parents=True, exist_ok=True)
        chart_id = uuid.uuid4().hex
        shutil.copyfile(png, charts_dir / f"{chart_id}.png")
        # How long a chart lives is decided by the saved chats (web.py), not here.
        return ChartResult(chart_id=chart_id)
