# The chart sandbox of the Visualization Specialist: Python, pandas and
# matplotlib, and nothing else. It runs with no network, a read-only root, no
# capabilities and as an unprivileged user — see crew/sandbox.py for the rest.
FROM python:3.13-slim
RUN pip install --no-cache-dir --only-binary=:all: --timeout 60 --retries 10 \
        "pandas>=2.2,<3" "matplotlib>=3.9,<4" \
 && useradd --uid 10001 --create-home sandbox
USER 10001
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl HOME=/tmp PYTHONDONTWRITEBYTECODE=1
WORKDIR /job
